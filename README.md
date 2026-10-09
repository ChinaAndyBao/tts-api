# Qwen3-TTS API 接口文档

模型常驻内存的 HTTP 服务。**所有合成延迟约 28x 慢于实时**（0.6B CPU）：5 秒音频 ≈ 2 分钟。
`/health` 免鉴权，其余接口需请求头 `X-API-Key`。

## 鉴权

```bash
KEY=$(grep TTS_API_KEYS /etc/tts-api.env | cut -d= -f2)
curl -H "X-API-Key: $KEY" ...
```

## 同步接口（连接保持到音频生成完）

| 方法 | 路径 | 入参 | 返回 |
|---|---|---|---|
| POST | `/v1/speak` | JSON: `text, speaker, language?` | audio/wav |
| POST | `/v1/clone` | JSON: `text, voice_id, language?` | audio/wav |
| POST | `/v1/clone/upload` | multipart: `text, ref_audio, ref_text, language?` | audio/wav |

```bash
curl -X POST http://<IP>:9898/v1/speak -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好呀","speaker":"vivian","language":"Chinese"}' \
  --output out.wav
```

`speaker` 取值见 `GET /v1/voices`（9 个自带音色，大小写不敏感）；`language` 不传自动检测。
`instruct` 参数已移除：0.6B 模型不支持（传入返回 422），换 1.7B-CustomVoice 后可恢复。

## 克隆音色（两步，特征只算一次）

```bash
# 1. 注册：上传参考音频 + ref_text（必填，参考音频的转写文本），拿 voice_id
curl -X POST http://<IP>:9898/v1/voices -H "X-API-Key: $KEY" \
  -F ref_audio=@ref.wav -F ref_text="参考音频的文本" -F name="我的音色"
# => {"voice_id": "abc123", "name": "我的音色"}

# 2. 用 voice_id 合成（不再传音频）
curl -X POST http://<IP>:9898/v1/clone -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"这是克隆声音","voice_id":"abc123"}' --output clone.wav
```

一次性临时克隆（不注册）用 `/v1/clone/upload`。voice_id 持久化在服务端，重启不丢。

## 异步任务（适合前端/移动端，不挂长连接）

```bash
# 提交，秒回
curl -X POST http://<IP>:9898/v1/tasks -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"type":"speak","text":"你好","speaker":"ryan"}'
# => {"task_id":"...","status":"queued","poll":"/v1/tasks/..."}

# 轮询状态（status: queued | running | succeeded | failed）
curl -H "X-API-Key: $KEY" http://<IP>:9898/v1/tasks/<task_id>

# 完成后取音频
curl -H "X-API-Key: $KEY" http://<IP>:9898/v1/tasks/<task_id>/audio --output t.wav
```

`type=clone` 时改传 `{"type":"clone","text":"...","voice_id":"abc123"}`。
任务 24 小时后自动清理。

## 其他

- `GET /v1/voices` — 自带音色列表、语种列表、已注册克隆音色
- `GET /health` — 健康检查

## 运维

```bash
systemctl status tts-api      # 状态
systemctl restart tts-api     # 重启（改配置后）
journalctl -u tts-api -f      # 日志
```

配置在 `/etc/tts-api.env`（API Key、并发数）。并发 `TTS_MAX_CONCURRENT=4`（压测实测最优：与 8 槽吞吐相同但延迟最低、更省内存）。
