"""Qwen3-TTS HTTP API 服务（模型常驻内存）。

  POST /v1/speak          自带音色同步合成      (JSON → 文件路径)
  POST /v1/clone          克隆音色同步合成      (JSON, voice_id → 文件路径)
  POST /v1/clone/upload   一次性克隆合成        (multipart: ref_audio → 文件路径)
  POST /v1/voices         注册克隆音色          (multipart: ref_audio)
  DELETE /v1/voices/{id}  删除克隆音色
  POST /v1/tasks          异步任务 speak/clone  (JSON)
  GET  /v1/tasks/{id}     任务状态
  GET  /v1/tasks/{id}/audio  任务音频
  GET  /v1/voices         音色/语种列表
  GET  /health            健康检查（免鉴权）
"""
from __future__ import annotations

import io
import json
import os
import pickle
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# 第三方依赖
#   numpy     —— 模型返回的音频是 np.ndarray，做幅度裁剪和 int16 量化
#   soundfile —— 把 PCM 写成 WAV 字节流（等价于写 wav 文件，只是落在内存）
#   torch     —— 显式导入：qwen_tts 的 VoiceClonePromptItem 含 torch.Tensor，
#                voices/store.pkl 里 pickle 的就是这些张量，不 import torch 会反序列化失败
#   qwen_tts  —— Qwen3-TTS 官方推理封装，本服务唯一的「模型能力」来源
# ---------------------------------------------------------------------------
import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from qwen_tts import Qwen3TTSModel

# ===========================================================================
# 路径与常量（全部硬编码，换环境必须改）
# ===========================================================================
# 服务目录（代码、任务音频 tasks/、音色缓存 voices/ 都在其下）
BASE_DIR = Path("/root/tts-api")
# 模型权重根目录，下面放两个模型子目录（见 lifespan）
MODELS_DIR = Path("/root/models")
# 异步任务合成结果（{task_id}.wav）+ 上传参考音频的临时文件（upload_*.wav）
TASKS_DIR = BASE_DIR / "tasks"
# 克隆音色目录（manifest.json 清单 + <id>.wav 参考音频 + <id>.pkl prompt 缓存）
VOICES_DIR = BASE_DIR / "voices"
# 启动即建目录，避免首次写文件时报 FileNotFoundError
TASKS_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# 鉴权 key：环境变量 TTS_API_KEYS，逗号分隔可配多个
# 模块级常量，import 时一次性读取 → 改密钥后必须重启才生效
API_KEYS = {k.strip() for k in os.environ.get("TTS_API_KEYS", "").split(",") if k.strip()}
# 同时控制两处并发（见下文 infer_slots 与 executor）：默认 4
# 压测结论：4 槽与 8 槽吞吐相同，但延迟更低、更省内存
MAX_CONCURRENT = int(os.environ.get("TTS_MAX_CONCURRENT", "4"))
# 任务保留 24 小时，超时的记录与 wav 一起清掉
TASK_TTL_SEC = 24 * 3600
# 任务表持久化文件（json）；音色清单同理
TASKS_STORE = TASKS_DIR / "tasks.json"
MANIFEST_STORE = VOICES_DIR / "manifest.json"
# 任务队列上限（排队 + 执行中），满则 POST /v1/tasks 返回 429
TASK_QUEUE_MAX = int(os.environ.get("TTS_TASK_QUEUE_MAX", "32"))
# 后台清理周期（秒）：过期任务 + 崩溃残留的 upload_*.wav
PRUNE_INTERVAL_SEC = 600

# ===========================================================================
# 进程内全局状态
#   服务是单进程多线程，模型实例全进程共用一份（常驻内存，避免每次请求都加载）
# ===========================================================================
# 自带音色合成模型（CustomVoice），lifespan 启动时加载
speak_model: Qwen3TTSModel | None = None
# 声音克隆模型（Base），lifespan 启动时加载
clone_model: Qwen3TTSModel | None = None
# 推理并发闸门：同时最多 MAX_CONCURRENT 路在跑模型，多余请求在 with 处排队
# 这是限流层 1（限的是「正在推理」的数量）
infer_slots = threading.Semaphore(MAX_CONCURRENT)
# 保护 voices 字典的锁（注册/读取时用）
voice_lock = threading.Lock()
# 已注册的克隆音色：voice_id -> entry（结构见 register_voice）
voices: dict[str, dict] = {}
# 异步任务表：task_id -> entry；持久化到 TASKS_STORE，重启不丢
tasks: dict[str, dict] = {}
# 保护 tasks 字典的锁
tasks_lock = threading.Lock()
# 异步任务执行器。这是限流层 2（限的是「同时执行的任务线程数」）
# 队列容量由 pending_slots 限住（排队+执行中 ≤ TASK_QUEUE_MAX，超了 429）
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
pending_slots = threading.BoundedSemaphore(TASK_QUEUE_MAX)


# ===========================================================================
# 模型加载（服务启动时执行一次，之后常驻内存）
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global speak_model, clone_model
    t0 = time.time()

    # ---- 模型 1/2：自带音色（CustomVoice）--------------------------------
    # Qwen3TTSModel.from_pretrained 的真实签名（qwen_tts/inference/qwen3_tts_model.py:83）：
    #     from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> Qwen3TTSModel
    #   - 内部依次 AutoConfig / AutoModel / AutoProcessor 注册并加载 "qwen3_tts" 这个自定义类型
    #   - **kwargs 原样透传给 AutoModel.from_pretrained，所以 device_map / dtype 都是 HF 的参数
    #   - 若模型目录里有 generate_config.json，会读成 generate_defaults（采样默认值）
    #   - 返回的是包装对象，含 model / processor / generate_defaults 三个成员
    #
    # device_map="cpu"      —— 纯 CPU 推理（本机无 GPU）；换 GPU 写 "cuda:0" 等
    # dtype=torch.float32   —— 用 fp32。CPU 上 bf16/fp16 往往更慢或不支持
    # 代价：0.6B fp32 + CPU → 合成约 28x 慢于实时（5 秒音频 ≈ 2 分钟）
    speak_model = Qwen3TTSModel.from_pretrained(
        str(MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        device_map="cpu", dtype=torch.float32)

    # ---- 模型 2/2：声音克隆（Base）--------------------------------------
    # 必须是 Base 模型：create_voice_clone_prompt / generate_voice_clone 内部会校验
    #   self.model.tts_model_type == "base"，不是就直接 raise ValueError
    # （对应地，generate_custom_voice 校验的是 == "custom_voice"）
    # 两个模型一起加载：启动慢一点，但换来请求路径上零加载开销
    clone_model = Qwen3TTSModel.from_pretrained(
        str(MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-Base"),
        device_map="cpu", dtype=torch.float32)

    # 模型就绪后回灌音色缓存/任务表，清掉崩溃残留的上传临时文件，
    # 再起后台清理线程（过期任务 + 孤儿文件，每 PRUNE_INTERVAL_SEC 一轮）
    _load_cached_voices()
    _load_tasks()
    _cleanup_upload_orphans()
    threading.Thread(target=_prune_loop, daemon=True).start()
    print(f"[startup] 2 models loaded in {time.time() - t0:.1f}s", flush=True)
    yield
    # 退出：不再接收新任务，但不强杀正在跑的推理
    executor.shutdown(wait=False)


app = FastAPI(title="Qwen3-TTS API", version="1.0", lifespan=lifespan)


# ===========================================================================
# 鉴权
# ===========================================================================
def require_api_key(x_api_key: str = Header("", alias="X-API-Key")):
    # 未配置 key 时报 500 而非 401：是配置缺失而非鉴权失败
    # （/health 免鉴权，用 api_keys_configured 字段暴露该盲区）
    if not API_KEYS:
        raise HTTPException(500, "TTS_API_KEYS 未配置")
    if x_api_key not in API_KEYS:
        raise HTTPException(401, "无效的 API Key")


# ---------- 音色缓存（create_voice_clone_prompt 结果持久化） ----------
# 为什么要持久化：create_voice_clone_prompt 要跑 speech_tokenizer.encode +
# extract_speaker_embedding，是整条链路里最贵的一步。缓存下来后，后续
# /v1/clone 只需一次 generate_voice_clone，省掉每次的特征提取。
#
# 存储布局（VOICES_DIR/）——prompt 缓存只是缓存，参考音频才是源数据：
#   manifest.json   清单：voice_id -> {name, ref_text, created_at, has_ref}
#   <id>.wav        注册时的参考音频（缓存失效时靠它重建 prompt）
#   <id>.pkl        prompt 缓存（含 torch.Tensor，与库版本相关）——坏了可重建

def _load_cached_voices():
    # 兼容旧版单文件 store.pkl：迁移成新布局后改名留档
    legacy = VOICES_DIR / "store.pkl"
    if legacy.exists() and not MANIFEST_STORE.exists():
        try:
            for vid, e in pickle.loads(legacy.read_bytes()).items():
                (VOICES_DIR / f"{vid}.pkl").write_bytes(pickle.dumps(e["prompt"]))
                voices[vid] = {"name": e["name"], "ref_text": e.get("ref_text"),
                               "created_at": e["created_at"], "has_ref": False,
                               "prompt": e["prompt"]}
            _persist_voices()
            legacy.rename(legacy.with_name("store.pkl.migrated"))
            print(f"[startup] migrated {len(voices)} voices from legacy store.pkl", flush=True)
        except Exception as e:
            print(f"[warn] legacy store.pkl 迁移失败: {e}", flush=True)
        return
    if not MANIFEST_STORE.exists():
        return
    try:
        manifest = json.loads(MANIFEST_STORE.read_bytes())
    except Exception as e:
        print(f"[warn] manifest 加载失败: {e}", flush=True)
        return
    for vid, meta in manifest.items():
        entry = dict(meta)
        pkl = VOICES_DIR / f"{vid}.pkl"
        try:
            entry["prompt"] = pickle.loads(pkl.read_bytes())
        except Exception:
            # prompt 缓存失效（如 torch 升级）：有参考音频就重建，没有就跳过
            ref = VOICES_DIR / f"{vid}.wav"
            if ref.exists():
                print(f"[startup] rebuild prompt for voice {vid}", flush=True)
                entry["prompt"] = clone_model.create_voice_clone_prompt(
                    str(ref), ref_text=meta.get("ref_text"))
                pkl.write_bytes(pickle.dumps(entry["prompt"]))
            else:
                print(f"[warn] voice {vid} prompt 失效且无参考音频，跳过", flush=True)
                continue
        voices[vid] = entry
    print(f"[startup] loaded {len(voices)} cached voices", flush=True)


def _persist_voices():
    # 先在锁内拷快照，锁外写盘；manifest 与每个 prompt 缓存都是原子写
    # （tmp 名带 uuid 防并发互踩，os.replace 同文件系统内原子）
    with voice_lock:
        data = dict(voices)
    manifest = {}
    for vid, e in data.items():
        manifest[vid] = {"name": e["name"], "ref_text": e.get("ref_text"),
                         "created_at": e["created_at"], "has_ref": e.get("has_ref", False)}
        if "prompt" in e:
            p = VOICES_DIR / f"{vid}.pkl"
            tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")
            tmp.write_bytes(pickle.dumps(e["prompt"]))
            os.replace(tmp, p)
    tmp = MANIFEST_STORE.with_name(f"{MANIFEST_STORE.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(json.dumps(manifest, ensure_ascii=False).encode())
    os.replace(tmp, MANIFEST_STORE)


def get_voice_prompt(voice_id: str):
    with voice_lock:
        entry = voices.get(voice_id)
    if entry is None:
        raise HTTPException(404, f"未知 voice_id: {voice_id}")
    # prompt = create_voice_clone_prompt 的返回值（List[VoiceClonePromptItem]）
    # 直接喂给 generate_voice_clone 的 voice_clone_prompt 参数
    return entry["prompt"]


# ---------- 工具 ----------

def wav_response(wav: np.ndarray, sr: int) -> Response:
    # 模型返回的 wav 是 float32，取值范围 [-1.0, 1.0]
    # 先 clip 防越界，再乘 32767 量化成 16bit PCM（WAV 通用格式）
    pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return Response(
        content=buf.getvalue(), media_type="audio/wav",
        # 顺手把时长塞进响应头，前端可直接显示进度/时长，不必解析 WAV
        headers={"X-Audio-Duration-Sec": f"{len(wav) / sr:.2f}"})


def _file_result(req_type: str, text: str, wavs, sr) -> dict:
    """合成结果落盘 tasks/<id>.wav 并登记为已完成任务，返回文件路径信息。

    同步接口不再直接回音频流：文件写在 tasks/（systemd 版即 OSS 挂载点；
    K8s 版由 sidecar 上传 OSS），调用方拿 file_name 到对应 OSS 目录取文件，
    或用 audio_url 走 HTTP 下载兜底。"""
    task_id = uuid.uuid4().hex
    path = TASKS_DIR / f"{task_id}.wav"
    path.write_bytes(wav_response(wavs[0], sr).body)
    duration_sec = round(len(wavs[0]) / sr, 2)
    with tasks_lock:
        tasks[task_id] = {"task_id": task_id, "type": req_type, "status": "succeeded",
                          "text": text[:80], "created_at": time.time(),
                          "finished_at": time.time(), "duration_sec": duration_sec}
    _persist_tasks()
    return {"task_id": task_id,
            "file_name": f"{task_id}.wav",
            "file_path": str(path),
            "duration_sec": duration_sec,
            "audio_url": f"/v1/tasks/{task_id}/audio"}


def canonical_speaker(name: str) -> str:
    # get_supported_speakers()（qwen_tts/inference/qwen3_tts_model.py:842）
    #   -> Optional[List[str]]：已排序、且全部转小写的官方音色名；模型不支持则返回 None
    # 名称校验必须先做：generate_custom_voice 内部也会校验，但那时已经进了推理段
    supported = speak_model.get_supported_speakers() or []
    for s in supported:
        if s.lower() == name.lower():
            # 返回 supported 里的写法（已是小写），保证传给模型的一定是合法值
            return s
    raise HTTPException(422, f"未知音色 {name!r}，可选: {supported}")


# ---------- 入参前置校验（把模型层的隐性约束提前暴露成 422） ----------

def _reject_instruct(instruct: str | None):
    # 为什么要有这个函数：
    #   qwen3_tts_model.py:799 里 `if self.model.tts_model_size in "0b6": instruct = None`
    #   —— 0.6B 模型不支持 instruct，源码会静默把它丢掉。
    #   本服务用的正是 Qwen3-TTS-12Hz-0.6B-CustomVoice（0.6B），
    #   所以「传了 instruct 却毫无效果」是最坑的失败模式。
    #   这里改成显式 422，让调用方立刻知道参数不被支持，而不是以为语气生效了。
    #   仅 1.7B-CustomVoice 支持 instruct，换模型后可移除本校验。
    if instruct:
        raise HTTPException(422, "0.6B 模型不支持 instruct（仅 1.7B-CustomVoice 支持）")


def _require_ref_text(ref_text: str):
    # 为什么要有这个函数：
    #   create_voice_clone_prompt 默认 x_vector_only_mode=False（ICL 模式），
    #   qwen_tts/inference/qwen3_tts_model.py:435 会对空 ref_text 直接：
    #       raise ValueError("ref_text is required when x_vector_only_mode=False (ICL mode)")
    #   与其让它在模型内部炸成 500，不如在接口层提前判成 422，语义更准。
    #   （若想让 ref_text 变可选，需显式传 x_vector_only_mode=True，只用说话人向量克隆，
    #     但克隆相似度会下降 —— 当前设计选择是保持 ICL 模式、强制 ref_text）
    if not ref_text.strip():
        raise HTTPException(422, "ref_text 不能为空（ICL 模式必填）")


# ---------- 同步接口 ----------

class SpeakRequest(BaseModel):
    text: str
    speaker: str
    language: str | None = None
    # 字段保留在契约里但永远被 _reject_instruct 拒掉：
    # 目的是「报错」而非「静默忽略」，避免调用方以为语气生效
    instruct: str | None = None


class CloneRequest(BaseModel):
    text: str
    voice_id: str
    language: str | None = None


@app.post("/v1/speak", dependencies=[Depends(require_api_key)])
def speak(req: SpeakRequest):
    # 0.6B 不支持 instruct → 显式 422（见 _reject_instruct）
    _reject_instruct(req.instruct)
    # 音色名归一（大小写不敏感）；找不到直接 422 并回显可选列表
    speaker = canonical_speaker(req.speaker)
    # 限流：同时最多 MAX_CONCURRENT 路在推理，多余请求在此阻塞排队
    # 本端点是 def（同步）→ 跑在 FastAPI 线程池里，排队时不会卡住事件循环
    with infer_slots:
        # ======================== 模型调用 ========================
        # speak_model.generate_custom_voice 的真实签名
        # （qwen_tts/inference/qwen3_tts_model.py:732）：
        #     generate_custom_voice(
        #         text, speaker, language=None, instruct=None,
        #         non_streaming_mode=True, **kwargs
        #     ) -> Tuple[List[np.ndarray], int]
        #
        # 前置校验（在模型内部）：
        #   * self.model.tts_model_type 必须 == "custom_voice"，否则 raise ValueError
        #   * speaker 会对照 get_supported_speakers() 校验（大小写不敏感）
        #   * language=None 时内部按 "Auto" 处理（自动检测语种）
        #
        # 各参数：
        #   text       要合成的文本（可传 list 批量；本服务每次只传一条）
        #   speaker    canonical_speaker 归一后的官方音色名
        #   language   语种名，None = 自动检测
        #   instruct  本服务不传——接口层已 422 拒掉（0.6B 会静默丢弃，见 _reject_instruct）
        #   non_streaming_mode ⚠️ 不是「流式输出」开关
        #      源码 docstring 明确：该参数为 false 时也只是「模拟流式文本输入」，
        #      并不开启真正的流式输入或流式生成。
        #      无论 true/false，返回值都是完整的 List[np.ndarray]。
        #   **kwargs   还可直通 HuggingFace generate() 的采样参数，例如
        #              do_sample / top_k / top_p / temperature /
        #              repetition_penalty / max_new_tokens
        #              —— 本服务没有暴露它们，用的是模型 generate_config.json 里的默认值
        #
        # 返回值 Tuple[List[np.ndarray], int] = (wavs, sr)：
        #   wavs : list[np.ndarray]，float32，取值范围 [-1.0, 1.0]，长度=len(text)
        #   sr   : 采样率（int），随模型一起固定
        # 内部带 @torch.no_grad()，不会构建反向图
        wavs, sr = speak_model.generate_custom_voice(
            text=req.text, speaker=speaker,
            language=req.language,
            non_streaming_mode=True)
    # 本服务一次只合成一条 → 取 wavs[0]；落盘 tasks/ 并返回文件路径
    return _file_result("speak", req.text, wavs, sr)


@app.post("/v1/clone", dependencies=[Depends(require_api_key)])
def clone(req: CloneRequest):
    # 从缓存里取出注册时算好的 prompt；voice_id 不存在直接 404
    # prompt = List[VoiceClonePromptItem]，由 create_voice_clone_prompt 产出
    prompt = get_voice_prompt(req.voice_id)
    with infer_slots:
        # ======================== 模型调用 ========================
        # clone_model.generate_voice_clone 的真实签名
        # （qwen_tts/inference/qwen3_tts_model.py:470）：
        #     generate_voice_clone(
        #         text, language=None,
        #         ref_audio=None, ref_text=None, x_vector_only_mode=False,
        #         voice_clone_prompt=None, non_streaming_mode=False, **kwargs
        #     ) -> Tuple[List[np.ndarray], int]
        #
        # 提供参考音色有两种方式（二选一）：
        #   A. 传 (ref_audio, ref_text) → 方法内部现算 prompt（每次都付特征提取的开销）
        #   B. 传 voice_clone_prompt   → 直接用预建好的 prompt（本服务走这条，省掉 A 的开销）
        # 本方法同样校验 self.model.tts_model_type == "base"
        #
        # 各参数：
        #   text               要合成的文本
        #   language           语种，None = 自动检测
        #   voice_clone_prompt 两种形态都接受：
        #                      Union[Dict[str, Any], List[VoiceClonePromptItem]]
        #                      本服务传的是后者（create_voice_clone_prompt 的返回值）
        #   non_streaming_mode 默认 False；本服务显式传 True
        #                      （同 speak 的说明：不是流式输出开关）
        #   **kwargs           同样可直通 HF generate() 采样参数（本服务未暴露）
        #
        # 返回值同 generate_custom_voice：(wavs: List[np.ndarray], sr: int)
        #   注：克隆链路慢得多（实测 p50 204s vs 78s）——
        #       Base 模型要额外处理 ref_code + 说话人向量
        wavs, sr = clone_model.generate_voice_clone(
            text=req.text, language=req.language,
            voice_clone_prompt=prompt, non_streaming_mode=True)
    return _file_result("clone", req.text, wavs, sr)


@app.post("/v1/clone/upload", dependencies=[Depends(require_api_key)])
def clone_upload(
        text: str = Form(...),
        ref_audio: UploadFile = File(...),
        ref_text: str = Form(...),
        language: str = Form(None)):
    # 本端点是 def（同步）→ 跑在 FastAPI 线程池，模型推理不会卡住事件循环
    # （旧版是 async def + 同步模型调用，会把整个 loop 堵死，连 /health 都无响应）
    # ref_text 用 Form(...) 必填：ICL 模式下模型强依赖它（见 _require_ref_text）
    _require_ref_text(ref_text)
    # 模型只认本地 wav 路径 / URL / base64 / (ndarray, sr)，不认 UploadFile 对象
    # → 先把上传内容落盘成临时 wav，用完删掉
    tmp = TASKS_DIR / f"upload_{uuid.uuid4().hex}.wav"
    # ref_audio.file 是 SpooledTemporaryFile，同步读即可（本端点是 def，没有 await）
    # 依赖 starlette 在 multipart 解析后已把指针 seek(0)（formparsers.py 约 :289）
    tmp.write_bytes(ref_audio.file.read())
    try:
        with infer_slots:
            # ============ 模型调用 1/2：建 prompt ============
            # clone_model.create_voice_clone_prompt 的真实签名
            # （qwen_tts/inference/qwen3_tts_model.py:356）：
            #     create_voice_clone_prompt(
            #         ref_audio, ref_text=None, x_vector_only_mode=False
            #     ) -> List[VoiceClonePromptItem]
            #
            # ref_audio 支持：str（本地 wav 路径 / URL / base64）、(np.ndarray, sr)、或它们的 list
            # 本服务传的是本地临时文件路径 str(tmp)
            #
            # ref_text  已在接口层用 _require_ref_text 挡掉空值
            #           （模型侧 qwen3_tts_model.py:435 对空 ref_text 会 raise ValueError）
            # x_vector_only_mode  用默认 False = ICL 模式（ref_code + ref_text 做上下文，
            #                     克隆相似度更高）。传 True 则只用说话人向量，ref_text 可省。
            #
            # 返回 List[VoiceClonePromptItem]，元素字段（qwen3_tts_model.py:41）：
            #     ref_code           Optional[torch.Tensor]  # (T, Q) 或 (T,)
            #     ref_spk_embedding  torch.Tensor            # (D,)
            #     x_vector_only_mode bool
            #     icl_mode           bool
            #     ref_text           Optional[str]
            # 内部做的事：speech_tokenizer.encode(参考音频) 抽 ref_code；
            #             extract_speaker_embedding(重采样到 24k 的音频) 抽说话人向量
            prompt = clone_model.create_voice_clone_prompt(str(tmp), ref_text=ref_text)
            # ============ 模型调用 2/2：合成 ============
            # voice_clone_prompt 直接喂上面现算的 prompt，参数含义见 /v1/clone
            wavs, sr = clone_model.generate_voice_clone(
                text=text, language=language,
                voice_clone_prompt=prompt, non_streaming_mode=True)
    finally:
        # 崩溃残留的 upload_*.wav 由启动/定期清理兜底（_cleanup_upload_orphans）
        tmp.unlink(missing_ok=True)
    return _file_result("clone", text, wavs, sr)


# ---------- 克隆音色注册 ----------

@app.get("/v1/voices", dependencies=[Depends(require_api_key)])
def list_voices():
    with voice_lock:
        registered = [
            {"voice_id": vid, "name": e["name"], "created_at": e["created_at"]}
            for vid, e in voices.items()]
    return {
        # 两个都是 qwen_tts 的能力查询接口（qwen3_tts_model.py:842 / :861）
        #   -> Optional[List[str]]，返回「已排序 + 全小写」的列表，模型不支持则为 None
        # 注：返回值全小写，与官方文档的大小写可能不一致
        "speakers": speak_model.get_supported_speakers(),
        "languages": speak_model.get_supported_languages(),
        "clone_voices": registered,
    }


@app.post("/v1/voices", dependencies=[Depends(require_api_key)])
def register_voice(
        ref_audio: UploadFile = File(...),
        ref_text: str = Form(...),
        name: str = Form("")):
    # def（同步）→ 跑线程池，不阻塞事件循环；ref_text 必填（ICL 模式）
    _require_ref_text(ref_text)
    voice_id = uuid.uuid4().hex[:12]
    # 参考音频永久留在 voices/<id>.wav（源数据，prompt 缓存坏了可据此重建）
    ref_path = VOICES_DIR / f"{voice_id}.wav"
    ref_path.write_bytes(ref_audio.file.read())
    try:
        with infer_slots:
            # ============ 模型调用：抽取并固化音色特征 ============
            # create_voice_clone_prompt(ref_audio, ref_text=None, x_vector_only_mode=False)
            #   -> List[VoiceClonePromptItem]
            # 这是整条链路最贵的一步（音频编码 + 说话人向量提取），
            # 做完缓存进 voices，之后 /v1/clone 就能反复复用，不用重算
            # 参数与返回值细节见 clone_upload 里的同一调用
            prompt = clone_model.create_voice_clone_prompt(str(ref_path), ref_text=ref_text)
    except Exception:
        ref_path.unlink(missing_ok=True)   # 建 prompt 失败就不留半成品
        raise
    entry = {
        # 未指定 name 时用 voice_id 前 6 位自动生成
        "name": name or f"voice-{voice_id[:6]}",
        # ★ 核心字段：List[VoiceClonePromptItem]，内含 torch.Tensor
        #   直接被 /v1/clone 与 _run_task 的 clone 分支消费
        "prompt": prompt,
        "ref_text": ref_text,
        "created_at": time.time(),
        "has_ref": True,
    }
    with voice_lock:
        voices[voice_id] = entry
    _persist_voices()          # manifest + prompt 缓存立刻落盘，重启不丢
    return {"voice_id": voice_id, "name": entry["name"]}


@app.delete("/v1/voices/{voice_id}", dependencies=[Depends(require_api_key)])
def delete_voice(voice_id: str):
    with voice_lock:
        if voice_id not in voices:
            raise HTTPException(404, f"未知 voice_id: {voice_id}")
        voices.pop(voice_id)
    (VOICES_DIR / f"{voice_id}.wav").unlink(missing_ok=True)
    (VOICES_DIR / f"{voice_id}.pkl").unlink(missing_ok=True)
    _persist_voices()
    return {"deleted": voice_id}


# ---------- 异步任务 ----------

class TaskRequest(BaseModel):
    type: str                      # "speak" | "clone"
    text: str
    speaker: str | None = None
    voice_id: str | None = None
    language: str | None = None
    # 同 SpeakRequest.instruct：保留字段但永远被 _reject_instruct 拒掉
    instruct: str | None = None


def _load_tasks():
    # 启动时回灌任务表；崩溃时进行中的任务无法续跑，如实标记 failed
    if not TASKS_STORE.exists():
        return
    try:
        tasks.update(json.loads(TASKS_STORE.read_bytes()))
        for t in tasks.values():
            if t["status"] in ("queued", "running"):
                t["status"] = "failed"
                t["error"] = "service restarted during processing"
        print(f"[startup] loaded {len(tasks)} tasks", flush=True)
        _persist_tasks()   # 把崩溃残留的 running 状态固化为 failed
    except Exception as e:
        print(f"[warn] 任务表加载失败: {e}", flush=True)


def _persist_tasks():
    with tasks_lock:
        data = dict(tasks)
    tmp = TASKS_STORE.with_name(f"{TASKS_STORE.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(json.dumps(data, ensure_ascii=False).encode())
    os.replace(tmp, TASKS_STORE)


def _cleanup_upload_orphans(age_sec: float = 0):
    # age_sec=0 全清（启动时用，此时没有在途请求）；定期清理只动 1h 前的
    now = time.time()
    for f in TASKS_DIR.glob("upload_*.wav"):
        if age_sec <= 0 or now - f.stat().st_mtime > age_sec:
            f.unlink(missing_ok=True)


def _prune_loop():
    # 后台定时清理：过期任务 + 1h 前的上传临时文件
    while True:
        time.sleep(PRUNE_INTERVAL_SEC)
        try:
            _prune_tasks()
            _cleanup_upload_orphans(age_sec=3600)
        except Exception as e:
            print(f"[warn] 定期清理失败: {e}", flush=True)


def _prune_tasks():
    now = time.time()
    with tasks_lock:
        stale = [tid for tid, t in tasks.items()
                 if now - t["created_at"] > TASK_TTL_SEC]
        for tid in stale:
            tasks.pop(tid, None)
    # 删 wav 放在锁外，避免持锁做磁盘 IO
    for tid in stale:
        (TASKS_DIR / f"{tid}.wav").unlink(missing_ok=True)
    if stale:
        _persist_tasks()


def _run_task(task_id: str, req: TaskRequest):
    # 在 executor 线程里执行，先标记 running
    with tasks_lock:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["started_at"] = time.time()
    _persist_tasks()
    try:
        # 合成（失败自动重试 1 次，瞬时抖动不直接判死）
        wavs = sr = None
        for attempt in (1, 2):
            try:
                # 同样走 infer_slots 限流；executor 的 max_workers 与它同为 MAX_CONCURRENT
                with infer_slots:
                    if req.type == "speak":
                        # ---- 模型调用：自带音色（参数含义见 /v1/speak）----
                        # 不传 instruct：create_task 已用 _reject_instruct 拦掉
                        # canonical_speaker 此处兜底校验（create_task 已提前校验过一次）
                        wavs, sr = speak_model.generate_custom_voice(
                            text=req.text, speaker=canonical_speaker(req.speaker),
                            language=req.language,
                            non_streaming_mode=True)
                    else:
                        # ---- 模型调用：克隆音色（参数含义见 /v1/clone）----
                        # prompt 由注册时算好并缓存，这里直接复用
                        wavs, sr = clone_model.generate_voice_clone(
                            text=req.text, language=req.language,
                            voice_clone_prompt=get_voice_prompt(req.voice_id),
                            non_streaming_mode=True)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"[warn] task {task_id} 合成失败，重试: {e}", flush=True)
        path = TASKS_DIR / f"{task_id}.wav"
        # 复用 wav_response 拿 WAV 字节（内部做 float32 → 16bit PCM）
        path.write_bytes(wav_response(wavs[0], sr).body)
        with tasks_lock:
            tasks[task_id].update(
                status="succeeded",
                finished_at=time.time(),
                # 音频时长（秒）= 样本数 / 采样率
                duration_sec=round(len(wavs[0]) / sr, 2))
        _persist_tasks()
    except Exception as e:
        # 失败记录进任务表（落盘）并打日志，便于排查
        print(f"[error] task {task_id} failed: {e}", flush=True)
        with tasks_lock:
            tasks[task_id].update(status="failed", error=str(e),
                                  finished_at=time.time())
        _persist_tasks()
    finally:
        pending_slots.release()


@app.post("/v1/tasks", dependencies=[Depends(require_api_key)])
def create_task(req: TaskRequest):
    # 顺带触发一次清理（后台 prune_loop 也会定期清）
    _prune_tasks()
    if req.type == "speak":
        if not req.speaker:
            raise HTTPException(422, "speak 任务必须提供 speaker")
        canonical_speaker(req.speaker)          # 提前校验，同步报错
    elif req.type == "clone":
        if not req.voice_id:
            raise HTTPException(422, "clone 任务必须提供 voice_id")
        get_voice_prompt(req.voice_id)          # 提前校验
    else:
        raise HTTPException(422, "type 必须是 speak 或 clone")
    # speak 与 clone 都不支持 instruct（0.6B），统一在这里拒掉
    # 注意顺序：type/speaker/voice_id 先校验，instruct 后校验，
    #          所以「非法 type + 带 instruct」会先报 type 的错
    _reject_instruct(req.instruct)
    # 背压：排队+执行中的任务数超上限直接拒，避免内存无限堆积
    if not pending_slots.acquire(blocking=False):
        raise HTTPException(429, f"任务队列已满（上限 {TASK_QUEUE_MAX}），稍后重试")
    task_id = uuid.uuid4().hex
    with tasks_lock:
        tasks[task_id] = {
            "task_id": task_id, "type": req.type, "status": "queued",
            # 只留前 80 字，避免长文本把内存表撑大
            "text": req.text[:80], "created_at": time.time()}
    _persist_tasks()
    try:
        # 提交即返回（秒回），实际合成在 executor 线程里跑
        executor.submit(_run_task, task_id, req)
    except Exception:
        pending_slots.release()
        raise
    return {"task_id": task_id, "status": "queued",
            "poll": f"/v1/tasks/{task_id}"}


@app.get("/v1/tasks/{task_id}", dependencies=[Depends(require_api_key)])
def task_status(task_id: str):
    with tasks_lock:
        t = tasks.get(task_id)
    # 任务表已持久化，重启后仍在（崩溃时进行中的任务标记 failed）
    if t is None:
        raise HTTPException(404, f"未知任务: {task_id}")
    # 浅拷贝一份再补 audio_url，避免把 audio_url 写回共享的 tasks 字典
    out = dict(t)
    if t["status"] == "succeeded":
        out["audio_url"] = f"/v1/tasks/{task_id}/audio"
        out["file_name"] = f"{task_id}.wav"
    return out


@app.get("/v1/tasks/{task_id}/audio", dependencies=[Depends(require_api_key)])
def task_audio(task_id: str):
    with tasks_lock:
        t = tasks.get(task_id)
    if t is None:
        raise HTTPException(404, f"未知任务: {task_id}")
    # 未完成就取音频 → 409（区别于 404，方便前端判断是「没这个任务」还是「还没好」）
    if t["status"] != "succeeded":
        raise HTTPException(409, f"任务未完成: {t['status']}")
    data = (TASKS_DIR / f"{task_id}.wav").read_bytes()
    return Response(content=data, media_type="audio/wav",
                    headers={"X-Audio-Duration-Sec": str(t.get("duration_sec", ""))})


@app.get("/health")
def health():
    # 免鉴权探活接口。key 未配置时 status=degraded（其余接口会 500），监控盯 status 即可
    # status=degraded 表示服务活着但配置有问题，一眼可辨
    return {
        "status": "ok" if API_KEYS else "degraded",
        # 两个模型是否都加载成功（lifespan 跑完才为 True）
        "models_loaded": speak_model is not None and clone_model is not None,
        "api_keys_configured": bool(API_KEYS),
        "voices": len(voices),
        "tasks": len(tasks),
        # 队列水位（排队+执行中），接近 TASK_QUEUE_MAX 说明该扩容了
        "inflight": sum(1 for t in tasks.values()
                        if t["status"] in ("queued", "running")),
    }
