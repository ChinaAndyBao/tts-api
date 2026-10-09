# Qwen3-TTS API — Kubernetes 部署手册（UAT / 生产）

> 适用于 UAT/生产以「镜像打包 → K8s 发布」的交付方式。单机 systemd 版见 `DEPLOY.md`。
> 已为 K8s 处理三件事：**OSS 上传改 sidecar（免特权）**、**模型权重走 PVC（不进镜像）**、**探针/资源按压测数据配置**。

## 1. 与 systemd 版的差异

| 环节 | systemd 版 | K8s 版 |
|---|---|---|
| 进程托管 | systemd unit | Deployment（探针自动拉活） |
| OSS 音频上传 | rclone FUSE 挂载 tasks/ | **oss-uploader sidecar** 每 30s `rclone copy`（无需特权容器） |
| 模型权重 | 本地 /root/models | **PVC**（tts-data/models），initContainer 自动补齐 |
| 克隆音色缓存 store.pkl | 本地 voices/ | **PVC**（tts-data/voices），重启不丢 |
| API 密钥 | /etc/tts-api.env | **K8s Secret**（tts-api-secret） |
| SELinux 适配 | 启动脚本放 /usr/local/sbin | 不需要（容器内无此限制） |

> **为什么不用 rclone mount**：K8s 里 FUSE 挂载要求特权容器 + 宿主机 /dev/fuse，多数集群被安全策略禁止。
> sidecar 用 `rclone copy` 上传（`--min-age 30s` 防半个文件上传），OSS 端**只增不删**——音频留存归 OSS 侧管理。

## 2. 镜像构建

```bash
# 本地/CI 构建（模型不进镜像，镜像约 3GB）
docker build -t <REGISTRY>/tts-api:<TAG> .
docker push <REGISTRY>/tts-api:<TAG>
```

> 国内构建注意：Dockerfile 里 torch 走阿里云轮子源；CI 机器访问 PyPI 慢的话加 `--build-arg PIP_INDEX=...` 自行扩展。

## 3. 预置资源

### 3.1 密钥（勿提交真实值）

```bash
kubectl create secret generic tts-api-secret --from-literal=TTS_API_KEYS=<随机密钥>
kubectl create secret generic tts-oss-rclone --from-file=rclone.conf=<你的 rclone.conf>
```

模板见 `deploy/k8s/00-secret.yaml.example`（改好另存 secret.yaml 再 apply 也行）。

### 3.2 持久卷

```bash
kubectl apply -f deploy/k8s/10-pvc.yaml
```

**模型权重两种就位方式（二选一）：**

- **A. 预置（推荐，生产用）**：把 `/root/models` 三个仓库拷进 PVC 的 `models/` 目录（可用一次性 Job 挂 PVC 从 OSS/内网源拷）。initContainer 检测到权重已存在即跳过下载。
- **B. 自动下载（UAT/测试方便）**：PVC 留空，Pod 首次启动 initContainer 自动从 ModelScope 拉 4.3GB（要求集群可访问 ModelScope）。

### 3.3 部署清单按环境修改点

| 文件 | 改什么 |
|---|---|
| `20-deployment.yaml` | `image: <REGISTRY>/tts-api:<TAG>`（两处）、`OSS_TTS_DIR`（test-tts / uat-tts / pro-tts） |
| `40-ingress.yaml.example` | `host:` 改为环境域名，另存 ingress.yaml |
| `10-pvc.yaml` | `storageClassName` 按集群存储类 |

## 4. 部署

```bash
kubectl apply -f deploy/k8s/20-deployment.yaml
kubectl apply -f deploy/k8s/30-service.yaml
kubectl apply -f deploy/k8s/40-ingress.yaml        # 可选

kubectl rollout status deploy/tts-api
kubectl logs deploy/tts-api -c api -f               # 应看到 "2 models loaded in ~xs"
```

## 5. 验证

```bash
KEY=<TTS_API_KEYS 的值>

# ① 集群内健康检查
kubectl exec deploy/tts-api -c api -- curl -s http://127.0.0.1:9898/health
#   期望 {"status":"ok","models_loaded":true,"api_keys_configured":true,...}

# ② 集群外访问（Ingress 或 port-forward）
kubectl port-forward svc/tts-api 9898:9898 &
curl -X POST http://127.0.0.1:9898/v1/speak -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好","speaker":"vivian","language":"Chinese"}' --output /tmp/out.wav

# ③ 异步任务音频是否上 OSS（约 60s 内）
kubectl logs deploy/tts-api -c oss-uploader --tail=20
# 或在 OSS 控制台看 va-ai/<env>-tts/
```

## 6. 环境差异速查

| 项 | 测试 | UAT | 生产 |
|---|---|---|---|
| namespace | test | uat | prod |
| 镜像 tag | test-<build> | uat-<build> | vX.Y.Z（固定版本，勿用 latest） |
| `OSS_TTS_DIR` | `test-tts` | `uat-tts` | `pro-tts` |
| Ingress host | tts-api.test.xxx | tts-api.uat.xxx | tts-api.xxx |
| replicas | 1 | 1 | 1（多副本见 7.3） |
| 模型就位 | 方式 B 自动下载 | 方式 A 预置 | 方式 A 预置 |

## 7. 运维要点

### 7.1 发布/升级
- 策略 `Recreate`：新旧 Pod 不并存（模型盘 RWO 限制 + 推理进程不宜双开）
- 停机窗口约 1 分钟（新 Pod 启动加载模型 20~60s），调用方需容忍短暂停机或前端排队
- 发布后验证：`kubectl rollout status` + 第 5 节冒烟

### 7.2 日志与监控
```bash
kubectl logs deploy/tts-api -c api -f            # 服务日志
kubectl logs deploy/tts-api -c oss-uploader -f   # 上传日志
kubectl top pod -l app=tts-api                   # 资源水位
```
- 监控建议盯：`/health` 存活、任务失败率、`tasks` 队列长度、内存水位（压测峰值 17.8GB）

### 7.3 扩缩容
- 默认 **replicas: 1**：模型常驻内存（约 10GB/副本），多副本 = 内存×N 且各自排队
- 需要多副本时：存储换 **RWX**（NAS），并发总量 = `TTS_MAX_CONCURRENT × 副本数`（CPU 要同步加）
- 不建议 HPA 自动扩缩：模型冷启动 20~60s，扩容跟不上突发流量，靠异步任务队列削峰更稳

### 7.4 备份
- **voices 子卷**（克隆音色声纹缓存）：定期快照/备份，丢失 = 已注册克隆音色全部失效
- OSS 音频侧 retention 策略（sidecar 只增不删）

## 8. 关键注意事项

1. **网关/Ingress 超时必须 ≥300s**：同步 `/v1/speak` 长句可达 5 分钟，超时会 502（40-ingress.yaml.example 已配 600s）
2. **资源规格**：requests `4C/12Gi`、limits `16C/24Gi`（压测：CPU 峰值 ~11.8 核、内存峰值 17.8GB）。同节点别再堆重负载服务
3. **CPU 限流与 torch 线程**：若 limits.cpu 远小于宿主机核数，可给 api 容器加 `OMP_NUM_THREADS=8` 环境变量避免线程过订
4. **单副本语义**：任务表在内存里，Pod 重启后旧 task_id 查询返回 404（wav 已在 OSS，不受影响）
5. **Pod Security**：镜像默认 root 运行（代码硬编码 /root 路径）。若集群开启 restricted 策略，需改代码支持非 root 路径后再适配

## 9. 排错

| 现象 | 处理 |
|---|---|
| initContainer 卡在下载模型 | 集群到 ModelScope 不通 → 改用方式 A 预置权重 |
| `speech_tokenizer/config.json not found` | initContainer 的符号链接步骤没跑 → 查 model-init 日志 |
| oss-uploader 反复 `sync failed` | 查 secret tts-oss-rclone 的 rclone.conf 凭证/endpoint；OSS 目录名是否与环境一致 |
| 调用方偶发 502/504 | 网关读超时 <300s（见注意事项 1） |
| OOMKilled | limits.memory 不足 → 提到 28Gi，或降 TTS_MAX_CONCURRENT |
| Pod 起不来 `couldn't find key TTS_API_KEYS` | tts-api-secret 未建或键名拼错 |

## 10. 文件清单（deploy/k8s/）

```
00-secret.yaml.example   密钥模板（API Key + OSS 凭证）
10-pvc.yaml              持久卷（模型 + 音色缓存，subPath 分区）
20-deployment.yaml       主服务 + OSS 上传 sidecar + initContainer
30-service.yaml          Service（9898）
40-ingress.yaml.example  入口模板（网关超时 600s）
```
