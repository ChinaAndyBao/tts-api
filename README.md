# Qwen3-TTS API 接口文档

0.6B 双模型（自带音色 CustomVoice + 声音克隆 Base）常驻内存的 HTTP 服务。

> **延迟预期（重要）**：合成约 28x 慢于实时（CPU 推理）——2 字短句约 9s、17 字长句约 139s，
> 输出音频越长耗时越久。同步接口请设客户端超时 ≥300s；不耐等待走异步 `/v1/tasks`。
> 轻接口（/health、/v1/voices）毫秒级返回，不受合成负载影响。

---

## 通用约定

| 项 | 约定 |
|---|---|
| Base URL | `http://<IP>:9898` |
| 鉴权 | 除 `GET /health` 外全部接口需请求头 `X-API-Key: <密钥>` |
| 请求体 | JSON 接口用 `Content-Type: application/json`；上传接口用 `multipart/form-data` |
| 音频格式 | WAV（16bit PCM，24kHz 单声道） |
| 文本 | UTF-8，中文/英文/日韩德法俄葡西意 10 语种（`language` 不传自动检测） |

### 鉴权示例

```bash
KEY=$(grep TTS_API_KEYS /etc/tts-api.env | cut -d= -f2-)
curl -H "X-API-Key: $KEY" ...
```

### 错误响应格式

```json
{"detail": "错误说明"}
```

### 错误码速查

| HTTP | 含义 | 触发场景 |
|---|---|---|
| 401 | API Key 无效 | 缺 `X-API-Key` 头或密钥错误 |
| 404 | 资源不存在 | voice_id / task_id 不存在或已清理 |
| 409 | 状态冲突 | 任务未完成就取音频 |
| 422 | 参数不合法 | 音色名错误、ref_text 空、instruct 不支持、缺必填字段 |
| 429 | 任务队列满 | 异步任务排队+执行中超过上限（默认 32），稍后重试 |
| 500 | 服务配置缺失 | 服务端未配置 TTS_API_KEYS（运维问题，非调用方问题） |

### 文件存储说明

所有合成结果落盘 `tasks/<id>.wav`：
- **systemd 版**：`tasks/` 即 OSS 挂载点（`va-ai/<env>-tts/`），`file_name` 就是 OSS 对象名
- **K8s 版**：sidecar 每 30s 上传 OSS，`file_name` 即对象名
- `audio_url` 字段提供 HTTP 下载兜底（`/v1/tasks/{id}/audio`）
- 任务/文件 24 小时后自动清理

---

## 接口总览

| # | 方法 | 路径 | 用途 | 返回 |
|---|---|---|---|---|
| 1 | GET | `/health` | 健康检查（免鉴权） | JSON 状态 |
| 2 | GET | `/v1/voices` | 查音色/语种/已注册克隆音色 | JSON 列表 |
| 3 | POST | `/v1/speak` | 自带音色同步合成 | 文件路径 JSON |
| 4 | POST | `/v1/voices` | 注册克隆音色（算一次特征） | voice_id |
| 5 | DELETE | `/v1/voices/{voice_id}` | 删除克隆音色 | deleted |
| 6 | POST | `/v1/clone` | 克隆音色同步合成（用 voice_id） | 文件路径 JSON |
| 7 | POST | `/v1/clone/upload` | 一次性克隆（现场传参考音频） | 文件路径 JSON |
| 8 | POST | `/v1/tasks` | 提交异步合成任务 | task_id（秒回） |
| 9 | GET | `/v1/tasks/{task_id}` | 查询任务状态 | 状态 JSON |
| 10 | GET | `/v1/tasks/{task_id}/audio` | 下载任务音频 | audio/wav |

---

## 1. GET /health — 健康检查

**用途**：探活/监控。免鉴权；`status=degraded` 表示服务活着但密钥配置有问题（此时其余接口全 500）。

**请求**：无参数、无鉴权。

**响应 200**：

```json
{
  "status": "ok",
  "models_loaded": true,
  "api_keys_configured": true,
  "voices": 1,
  "tasks": 4,
  "inflight": 0
}
```

| 字段 | 说明 |
|---|---|
| `status` | `ok` 正常 / `degraded` 密钥未配置 |
| `models_loaded` | 双模型是否加载完成（启动后才 true） |
| `api_keys_configured` | 服务端是否配置了 API Key |
| `voices` | 已注册克隆音色数 |
| `tasks` | 任务表条数（含 24h 内的历史） |
| `inflight` | 队列水位（排队+执行中），接近队列上限（32）说明该分流了 |

```bash
curl http://<IP>:9898/health
```

---

## 2. GET /v1/voices — 查音色/语种/克隆音色

**用途**：调用方启动时拉取可选音色和语种（做前端下拉框）、查看已注册的克隆音色。毫秒级返回。

**请求**：无参数，需鉴权。

**响应 200**：

```json
{
  "speakers": ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"],
  "languages": ["auto", "chinese", "english", "japanese", "korean", "german", "french", "russian", "portuguese", "spanish", "italian"],
  "clone_voices": [
    {"voice_id": "db963c7defec", "name": "参考音色1008", "created_at": 1760000000.0}
  ]
}
```

| 字段 | 说明 |
|---|---|
| `speakers` | 9 个自带音色（传给 `/v1/speak` 的 speaker；全小写、大小写不敏感） |
| `languages` | 10 语种 + `auto` 自动检测（language 参数取值） |
| `clone_voices` | 已注册克隆音色（voice_id 供 `/v1/clone` 使用） |

```bash
curl -H "X-API-Key: $KEY" http://<IP>:9898/v1/voices
```

---

## 3. POST /v1/speak — 自带音色同步合成

**用途**：用官方 9 音色合成语音。同步阻塞到合成完成（短句约 9s，长句可达数分钟）。

**请求**（`application/json`，需鉴权）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | string | ✅ | 要合成的文本 |
| `speaker` | string | ✅ | 音色名，见 `GET /v1/voices`（大小写不敏感） |
| `language` | string | ❌ | 语种，不传自动检测 |
| `instruct` | string | ❌ | ⚠️ 当前部署为 0.6B 模型不支持，传入返回 422 |

**响应 200**：

```json
{
  "task_id": "3b2d5c5d8db44d598637f843ea56b27f",
  "file_name": "3b2d5c5d8db44d598637f843ea56b27f.wav",
  "file_path": "/root/tts-api/tasks/3b2d5c5d8db44d598637f843ea56b27f.wav",
  "duration_sec": 0.64,
  "audio_url": "/v1/tasks/3b2d5c5d8db44d598637f843ea56b27f/audio"
}
```

| 字段 | 说明 |
|---|---|
| `task_id` | 合成记录 id（可用于 /v1/tasks/{id} 查询） |
| `file_name` | **OSS 对象名**（调用方直接从 OSS 取文件） |
| `file_path` | 服务端落盘路径 |
| `duration_sec` | 音频时长（秒） |
| `audio_url` | HTTP 下载兜底地址 |

**错误**：422 音色名不存在（回显可选列表）/ 422 携带 instruct / 401 鉴权失败。

```bash
curl -X POST http://<IP>:9898/v1/speak -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好呀","speaker":"vivian","language":"Chinese"}'
```

---

## 4. POST /v1/voices — 注册克隆音色

**用途**：上传参考音频提取声纹特征，注册成可复用的克隆音色（特征只算这一次，后续 `/v1/clone` 直接用 voice_id，免重复传音频）。参考音频建议 5~30 秒清晰人声。

**请求**（`multipart/form-data`，需鉴权）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `ref_audio` | file | ✅ | 参考音频（wav/m4a/mp3 等 librosa 可读格式） |
| `ref_text` | string | ✅ | 参考音频的转写文本（ICL 模式强依赖，空值 422） |
| `name` | string | ❌ | 音色显示名，不传自动 `voice-xxxxxx` |

**响应 200**：

```json
{"voice_id": "abc123def456", "name": "我的音色"}
```

**错误**：422 缺 ref_text 或为空 / 422 音频无法解析 / 401 鉴权失败。

**持久化**：voice_id 落盘（manifest + 参考音频 + 特征缓存），服务重启不丢。
特征缓存损坏时可从参考音频自动重建；删除用 `DELETE /v1/voices/{id}`。

```bash
curl -X POST http://<IP>:9898/v1/voices -H "X-API-Key: $KEY" \
  -F ref_audio=@ref.wav -F ref_text="参考音频的转写文本" -F name="我的音色"
```

---

## 5. DELETE /v1/voices/{voice_id} — 删除克隆音色

**用途**：删除已注册的克隆音色（同步清理参考音频 wav 与特征缓存 pkl，manifest 更新）。

**请求**：路径参数 `voice_id`，需鉴权，无请求体。

**响应 200**：

```json
{"deleted": "abc123def456"}
```

**错误**：404 voice_id 不存在 / 401 鉴权失败。

```bash
curl -X DELETE http://<IP>:9898/v1/voices/abc123def456 -H "X-API-Key: $KEY"
```

---

## 6. POST /v1/clone — 克隆音色同步合成（voice_id）

**用途**：用已注册的克隆音色合成语音（参考音色特征已缓存，合成速度比 upload 略快；但克隆链路整体比自带音色慢 ~2x，Base 模型要额外处理声纹向量）。

**请求**（`application/json`，需鉴权）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | string | ✅ | 要合成的文本 |
| `voice_id` | string | ✅ | `POST /v1/voices` 注册返回的 id |
| `language` | string | ❌ | 语种，不传自动检测 |

**响应 200**：同 `/v1/speak`（文件路径 JSON）。

**错误**：404 voice_id 不存在 / 422 参数缺失 / 401 鉴权失败。

```bash
curl -X POST http://<IP>:9898/v1/clone -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"这是克隆声音","voice_id":"abc123def456","language":"Chinese"}'
```

---

## 7. POST /v1/clone/upload — 一次性克隆（不注册）

**用途**：现场传参考音频直接克隆合成，不落注册。适合试音、一次性场景；重复使用同一音色请走注册（省掉每次的特征提取）。

**请求**（`multipart/form-data`，需鉴权）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `text` | string | ✅ | 要合成的文本 |
| `ref_audio` | file | ✅ | 参考音频 |
| `ref_text` | string | ✅ | 参考音频的转写文本（必填，同注册） |
| `language` | string | ❌ | 语种，不传自动检测 |

**响应 200**：同 `/v1/speak`（文件路径 JSON）。

**错误**：422 缺 ref_text 或为空 / 422 音频无法解析 / 401 鉴权失败。

```bash
curl -X POST http://<IP>:9898/v1/clone/upload -H "X-API-Key: $KEY" \
  -F text="这是克隆声音" -F ref_audio=@ref.wav \
  -F ref_text="参考音频的转写文本" -F language=Chinese
```

---

## 8. POST /v1/tasks — 提交异步合成任务

**用途**：不挂长连接的合成。提交**秒回** task_id，后台排队合成；适合前端/移动端/批量提交。
队列上限默认 32（排队+执行中），满了返回 **429**。

**请求**（`application/json`，需鉴权）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | ✅ | `speak`（自带音色）或 `clone`（克隆） |
| `text` | string | ✅ | 要合成的文本 |
| `speaker` | string | type=speak 必填 | 音色名（同 /v1/speak） |
| `voice_id` | string | type=clone 必填 | 已注册克隆音色 id |
| `language` | string | ❌ | 语种，不传自动检测 |
| `instruct` | string | ❌ | ⚠️ 0.6B 不支持，传入返回 422 |

**响应 200**：

```json
{"task_id": "9667f3906e6b4c12b84c80c15719ab42", "status": "queued", "poll": "/v1/tasks/9667f3906e6b4c12b84c80c15719ab42"}
```

**错误**：422 缺 speaker/voice_id 或 type 非法 / 422 携带 instruct / **429 队列满**（稍后重试）/ 401。

**行为说明**：
- 提交即入队，秒回；合成在后台执行（单条耗时同同步接口）
- 失败自动重试 1 次；最终失败状态记 `failed` + `error` 字段
- 任务状态落盘，**服务重启不丢**（重启瞬间在途任务标记 failed）
- 24 小时后自动清理

```bash
# speak 任务
curl -X POST http://<IP>:9898/v1/tasks -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"type":"speak","text":"你好","speaker":"ryan"}'

# clone 任务
curl -X POST http://<IP>:9898/v1/tasks -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"type":"clone","text":"你好","voice_id":"abc123def456"}'
```

---

## 9. GET /v1/tasks/{task_id} — 查询任务状态

**用途**：轮询任务进度（建议间隔 5~15s）。完成后从响应取 `file_name` 去 OSS 取文件，或用 `audio_url` 下载。

**请求**：路径参数 `task_id`，需鉴权。

**响应 200（进行中）**：

```json
{
  "task_id": "9667f3906e6b4c12b84c80c15719ab42",
  "type": "speak",
  "status": "running",
  "text": "好",
  "created_at": 1791535415.62,
  "started_at": 1791535415.64
}
```

**响应 200（成功）**：

```json
{
  "task_id": "9667f3906e6b4c12b84c80c15719ab42",
  "type": "speak",
  "status": "succeeded",
  "text": "好",
  "created_at": 1791535415.62,
  "started_at": 1791535415.64,
  "finished_at": 1791535424.26,
  "duration_sec": 1.28,
  "audio_url": "/v1/tasks/9667f3906e6b4c12b84c80c15719ab42/audio",
  "file_name": "9667f3906e6b4c12b84c80c15719ab42.wav"
}
```

| 字段 | 说明 |
|---|---|
| `status` | `queued` 排队 → `running` 合成中 → `succeeded` / `failed` |
| `text` | 原文前 80 字（防长文本撑内存） |
| `error` | status=failed 时的错误原因 |
| `file_name` | 成功后：**OSS 对象名** |
| `audio_url` | 成功后：HTTP 下载地址 |

**错误**：404 task_id 不存在或超 24h 已清理 / 401。

```bash
curl -H "X-API-Key: $KEY" http://<IP>:9898/v1/tasks/9667f3906e6b4c12b84c80c15719ab42
```

---

## 10. GET /v1/tasks/{task_id}/audio — 下载任务音频

**用途**：HTTP 下载任务合成的音频（OSS 取文件的兜底通道）。响应头 `X-Audio-Duration-Sec` 带音频时长。

**请求**：路径参数 `task_id`，需鉴权。

**响应 200**：`audio/wav` 二进制（16bit PCM 24kHz）。

**错误**：404 task_id 不存在 / **409 任务未完成**（status 未到 succeeded）/ 401。

```bash
curl -H "X-API-Key: $KEY" \
  http://<IP>:9898/v1/tasks/9667f3906e6b4c12b84c80c15719ab42/audio \
  --output t.wav
```

---

## 一键自测脚本

复制整段执行（替换 IP），约 1 分钟跑完全部核心路径：

```bash
IP=192.168.10.133
KEY=$(grep TTS_API_KEYS /etc/tts-api.env | cut -d= -f2-)
BASE=http://$IP:9898

echo "① health:"; curl -s $BASE/health; echo
echo "② voices:"; curl -s -H "X-API-Key: $KEY" $BASE/v1/voices | head -c 300; echo
echo "③ speak:"; curl -s -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"text":"测试一下","speaker":"vivian"}' $BASE/v1/speak; echo
echo "④ 注册克隆音色:"
VID=$(curl -s -H "X-API-Key: $KEY" -F ref_audio=@ref.wav \
  -F ref_text="参考音频文本" -F name="自测音色" $BASE/v1/voices | python3 -c "import sys,json;print(json.load(sys.stdin)['voice_id'])")
echo "voice_id=$VID"
echo "⑤ 克隆合成:"; curl -s -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d "{\"text\":\"克隆测试\",\"voice_id\":\"$VID\"}" $BASE/v1/clone; echo
echo "⑥ 异步任务:"
TID=$(curl -s -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"type":"speak","text":"异步测试","speaker":"ryan"}' $BASE/v1/tasks | python3 -c "import sys,json;print(json.load(sys.stdin)['task_id'])")
echo "task_id=$TID，轮询:"; sleep 20; curl -s -H "X-API-Key: $KEY" $BASE/v1/tasks/$TID; echo
echo "⑦ 下载音频:"; curl -s -H "X-API-Key: $KEY" $BASE/v1/tasks/$TID/audio -o t.wav && ls -la t.wav
echo "⑧ 清理测试音色:"; curl -s -X DELETE -H "X-API-Key: $KEY" $BASE/v1/voices/$VID; echo
```

---

## 运维

```bash
systemctl status tts-api      # 状态
systemctl restart tts-api     # 重启（改配置后）
journalctl -u tts-api -f      # 日志
```

配置在 `/etc/tts-api.env`（API Key、并发数）：
- `TTS_MAX_CONCURRENT=4`（推理并发，压测实测最优）
- `TTS_TASK_QUEUE_MAX=32`（异步队列上限，满 429）
