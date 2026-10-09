# Qwen3-TTS API 部署手册（UAT / 生产）

> 在一台新的 Linux 服务器上从零部署 TTS API。已在 AlmaLinux 9.6（CPU-only，16 核 / 31GB）全流程验证。

## 1. 架构

```
调用方 ──HTTP(:9898, X-API-Key)──> FastAPI 服务 (tts-api)
                                    ├─ 模型常驻内存（0.6B CustomVoice + 0.6B Base）
                                    └─ 生成音频 → /root/tts-api/tasks/ ──rclone FUSE──> OSS va-ai/<env>-tts/
```

| systemd 服务 | 作用 | 启动顺序 |
|---|---|---|
| `oss-tts-mount.service` | rclone 挂载 OSS 音频目录 | 先 |
| `tts-api.service` | FastAPI API 服务 | 后（Before= 已声明） |

## 2. 环境要求

- Linux x86_64（AlmaLinux / RHEL 9 系已验证；Ubuntu 同理）
- CPU ≥ 8 核（**无需 GPU**；实测 16 核吞吐 ~4 句/分钟，见下文容量说明）
- 内存 ≥ 16GB（双模型常驻约 10GB + 并发缓冲 2~4GB）
- 磁盘 ≥ 20GB（模型权重 4.3GB + conda 环境约 3GB）
- 网络可达：ModelScope（仅下载模型时需要）、OSS endpoint（oss-cn-hangzhou.aliyuncs.com）
- 国内网络建议先配教育网镜像（见 3.1 / 3.2 的镜像源参数）

## 3. 安装步骤

### 3.1 系统包

```bash
# 音频工具 + FUSE（rclone 挂载依赖 fusermount3，缺了会 mount 失败）
dnf install -y sox fuse3        # RHEL 系；Ubuntu: apt install sox fuse3

# rclone（GitHub 国内直连慢，用阿里云 EPEL 镜像装）
curl -fL -o /tmp/rclone.rpm \
  https://mirrors.aliyun.com/epel/9/Everything/x86_64/Packages/r/rclone-1.74.3-1.el9.x86_64.rpm
rpm -Uvh /tmp/rclone.rpm
```

### 3.2 Python 环境（conda）

```bash
# Miniforge3（或已有 conda 均可）
bash Miniforge3-Linux-x86_64.sh -b -p /root/miniforge3
conda create -n qwen3-tts python=3.12 -y
conda config --set channel_alias https://mirrors.ustc.edu.cn/anaconda/cloud   # 国内镜像

# 依赖：先装 CPU 版 torch，再装其余（qwen-tts 及其依赖）
PIP=/root/miniforge3/envs/qwen3-tts/bin/pip
$PIP install torch==2.11.0+cpu torchaudio==2.11.0+cpu \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu/
$PIP install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

> `requirements.txt` 里 torch 用 `+cpu` 后缀（本机无 GPU 时省 ~2.5GB 无用的 CUDA 轮子）。
> 换 GPU 服务器时改为安装对应 CUDA 版本即可。

### 3.3 模型权重（约 4.3GB）

```bash
python3 deploy/download_models.py        # ModelScope 下载 3 个仓库到 /root/models/（带断点续传）
```

> ModelScope 将 speech_tokenizer 拆成独立仓库，必须挂进两个模型目录（HF 版内嵌，无需此步）：

```bash
for m in Qwen3-TTS-12Hz-0.6B-CustomVoice Qwen3-TTS-12Hz-0.6B-Base; do
  ln -sfn /root/models/Qwen3-TTS-Tokenizer-12Hz /root/models/$m/speech_tokenizer
done
```

### 3.4 服务代码与配置

```bash
# 代码就位（本仓库内容）
cp app.py README.md stress_test.py /root/tts-api/

# API 密钥（务必 chmod 600；可配多个，逗号分隔）
cp deploy/tts-api.env.example /etc/tts-api.env
chmod 600 /etc/tts-api.env
vim /etc/tts-api.env        # 改 TTS_API_KEYS

# 启动包装（SELinux 下 systemd 不能直接读 /root 下的文件，经 /usr/local/sbin 转接）
cp deploy/tts-api-run /usr/local/sbin/tts-api-run
chmod 755 /usr/local/sbin/tts-api-run
restorecon /usr/local/sbin/tts-api-run 2>/dev/null || true
```

### 3.5 systemd 服务

```bash
cp deploy/tts-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now tts-api
```

### 3.6 OSS 音频挂载（rclone）

```bash
mkdir -p /root/.config/rclone
# rclone.conf：参照 deploy/rclone.conf.example 填 AccessKey（与 Flask 服务共桶不同目录）
chmod 600 /root/.config/rclone/rclone.conf

cp deploy/oss-tts-mount.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now oss-tts-mount
```

> `oss-tts-mount.service` 中的 `aliyun-oss:va-ai/test-tts` **按环境修改**（见第 5 节表格）。
> 挂载参数要点：`--vfs-write-back 30s`（写后约 30 秒上传 OSS，生成型负载够用）；
> 音频目录 `/root/tts-api/tasks` 即挂载点，服务无需任何改动。

### 3.7 防火墙

```bash
firewall-cmd --permanent --add-port=9898/tcp && firewall-cmd --reload   # RHEL 系
```

## 4. 部署验证

```bash
KEY=$(grep TTS_API_KEYS /etc/tts-api.env | cut -d= -f2-)

# ① 健康检查（免鉴权）
curl http://127.0.0.1:9898/health
#   期望: {"status":"ok","models_loaded":true,"api_keys_configured":true,...}

# ② 同步合成（短句约 9s）
curl -X POST http://127.0.0.1:9898/v1/speak -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好","speaker":"vivian","language":"Chinese"}' --output /tmp/out.wav

# ③ 完整冒烟（合成 + 注册克隆音色 + 克隆合成，约 3~5 分钟）
/root/miniforge3/envs/qwen3-tts/bin/python deploy/smoke_test.py

# ④ 音频落 OSS（约 30s 后）
rclone ls aliyun-oss:va-ai/test-tts
```

## 5. 环境差异速查

| 项 | 测试 | UAT | 生产 |
|---|---|---|---|
| OSS 目录 | `va-ai/test-tts` | `va-ai/uat-tts` | `va-ai/pro-tts` |
| systemd 单元 | `oss-test-tts.service` | `oss-uat-tts.service` | `oss-pro-tts.service` |
| API 端口 | 9898 | 9898（或独立） | 9898（前置 Nginx/网关鉴权更佳） |
| `TTS_MAX_CONCURRENT` | 4 | 4 | 4 起步，按压测调（≥8 收益为零） |
| 模型权重 | 同一套（0.6B × 2） | 同左 | 同左 |

> 其余配置（代码、模型、密钥机制）三环境完全一致。目录结构保持 `/root/tts-api`、`/root/models` 不变可省去所有路径调整。

## 6. 运维手册

```bash
systemctl status tts-api oss-tts-mount      # 状态
systemctl restart tts-api                    # 改配置/代码后重启
journalctl -u tts-api -f                     # 服务日志
journalctl -u oss-tts-mount -f               # 挂载日志

# 密钥轮换：改 /etc/tts-api.env 后重启（改多个用逗号分隔）
systemctl restart tts-api

# 备份（重要）：克隆音色声纹缓存，丢了已注册音色全部失效
cp /root/tts-api/voices/store.pkl /root/backup/voices-store.pkl.$(date +%F)
```

**容量实测**（16 核 CPU，0.6B float32）：
- 吞吐硬极限 ~4 句/分钟（短句），输出越长越慢（约 28x 慢于实时）
- 同步接口建议调用方并发 ≤ 4、超时 ≥ 300s；更高并发走异步 `/v1/tasks`
- 轻接口（/v1/voices、/health）可达 187 RPS，不受合成负载影响

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `fusermount3: executable file not found` | 缺 fuse3：`dnf install -y fuse3` |
| systemd 启动报 `Failed to load environment files: Permission denied` | SELinux 禁止读 /root 下配置——env 必须放 `/etc/tts-api.env`，启动脚本走 `/usr/local/sbin`（3.4 已按此处理） |
| 模型加载报 `speech_tokenizer/config.json not found` | 3.3 的符号链接没做（ModelScope 专属问题） |
| 启动日志提示 flash-attn 未安装 | 正常（GPU 加速项），CPU 环境忽略 |
| EPEL 源下载极慢 | 用阿里云镜像直接下 rpm（3.1 已按此写） |
| `instruct` 参数返回 422 | 预期行为：0.6B 模型不支持 instruct（仅 1.7B-CustomVoice 支持） |
| `ref_text` 缺失返回 422 | 预期行为：克隆接口必填参考音频的转写文本 |

## 8. 组件清单

```
/root/tts-api/            服务目录
├── app.py                FastAPI 服务（模型常驻、同步/异步接口）
├── requirements.txt      Python 依赖
├── stress_test.py        压测工具（voices/speak/tasks/clone/mixed 五模式）
├── tasks/                生成音频（OSS 挂载点，勿手动清理也可）
├── voices/store.pkl      克隆音色缓存（含声纹向量，勿入 git、定期备份）
├── deploy/               部署产物
│   ├── tts-api.service   systemd 单元
│   ├── oss-tts-mount.service  OSS 挂载单元（按环境改目录）
│   ├── tts-api-run       启动包装（SELinux 合规）
│   ├── tts-api.env.example   环境变量模板
│   ├── rclone.conf.example   rclone 远端模板
│   ├── download_models.py    ModelScope 权重下载器
│   └── smoke_test.py         部署冒烟脚本
└── README.md             API 接口文档
```
