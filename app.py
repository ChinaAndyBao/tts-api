"""Qwen3-TTS HTTP API 服务（模型常驻内存）。

  POST /v1/speak          自带音色同步合成      (JSON)
  POST /v1/clone          克隆音色同步合成      (JSON, voice_id)
  POST /v1/clone/upload   一次性克隆合成        (multipart: ref_audio)
  POST /v1/voices         注册克隆音色          (multipart: ref_audio)
  POST /v1/tasks          异步任务 speak/clone  (JSON)
  GET  /v1/tasks/{id}     任务状态
  GET  /v1/tasks/{id}/audio  任务音频
  GET  /v1/voices         音色/语种列表
  GET  /health            健康检查（免鉴权）
"""
from __future__ import annotations

import io
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
#   torch     —— 显式导入：qwen_tts 内部的 VoiceClonePromptItem 含 torch.Tensor，
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
# 服务端真身目录。本机只是工作副本，实际部署在 /root/tts-api
BASE_DIR = Path("/root/tts-api")
# 模型权重根目录，下面放两个模型子目录（见 lifespan）
MODELS_DIR = Path("/root/models")
# 异步任务合成结果（{task_id}.wav）+ 上传参考音频的临时文件（upload_*.wav）
TASKS_DIR = BASE_DIR / "tasks"
# 克隆音色持久化目录（store.pkl）
VOICES_DIR = BASE_DIR / "voices"
# 启动即建目录，避免首次写文件时报 FileNotFoundError
TASKS_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# 鉴权 key：环境变量 TTS_API_KEYS，逗号分隔可配多个
# 注意：这是模块级常量，import 时一次性读取 → 改 /etc/tts-api.env 后必须重启才生效
API_KEYS = {k.strip() for k in os.environ.get("TTS_API_KEYS", "").split(",") if k.strip()}
# 同时控制两处并发（见下文 infer_slots 与 executor）：默认 4
# 压测结论：4 槽与 8 槽吞吐相同，但延迟更低、更省内存
MAX_CONCURRENT = int(os.environ.get("TTS_MAX_CONCURRENT", "4"))
# 任务保留 24 小时，超时的内存记录与 wav 一起清掉
TASK_TTL_SEC = 24 * 3600

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
# 保护 voices 字典的锁（注册/读取/落盘时用）
voice_lock = threading.Lock()
# 已注册的克隆音色：voice_id -> entry（结构见 register_voice）
voices: dict[str, dict] = {}
# 异步任务表：task_id -> entry。⚠️ 纯内存，服务重启全丢（wav 仍留在磁盘）
tasks: dict[str, dict] = {}
# 保护 tasks 字典的锁
tasks_lock = threading.Lock()
# 异步任务执行器。这是限流层 2（限的是「同时执行的任务线程数」）
# ⚠️ submit 的队列无界，高并发提交时任务会堆积成 queued，没有背压
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)


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

    # 模型就绪后回灌已注册的克隆音色（pickle 里存的是含 torch.Tensor 的 prompt）
    _load_cached_voices()
    print(f"[startup] 2 models loaded in {time.time() - t0:.1f}s", flush=True)
    yield
    # 退出：不再接收新任务，但不强杀正在跑的推理
    executor.shutdown(wait=False)


app = FastAPI(title="Qwen3-TTS API", version="1.0", lifespan=lifespan)


# ===========================================================================
# 鉴权
# ===========================================================================
def require_api_key(x_api_key: str = Header("", alias="X-API-Key")):
    # 未配置 key 时报 500 而非 401 —— ⚠️ 这是配置缺失，不是鉴权失败
    # 连带问题：/health 不走本依赖、永远 200，监控会误以为服务正常但所有接口全挂
    if not API_KEYS:
        raise HTTPException(500, "TTS_API_KEYS 未配置")
    if x_api_key not in API_KEYS:
        raise HTTPException(401, "无效的 API Key")


# ---------- 音色缓存（create_voice_clone_prompt 结果持久化） ----------
# 为什么要持久化：create_voice_clone_prompt 要跑 speech_tokenizer.encode +
# extract_speaker_embedding，是整条链路里最贵的一步。缓存下来后，后续
# /v1/clone 只需一次 generate_voice_clone，省掉每次的特征提取。

def _voices_store_path() -> Path:
    return VOICES_DIR / "store.pkl"


def _load_cached_voices():
    p = _voices_store_path()
    if p.exists():
        try:
            # store.pkl 内容 = dict[voice_id, entry]，entry["prompt"] 是
            #   List[VoiceClonePromptItem]，元素含 torch.Tensor（ref_code / ref_spk_embedding）
            # ⚠️ 因此与 qwen_tts / torch 版本强绑定：升级库后可能反序列化失败，
            #    那时所有已注册 voice_id 会一起失效（下面只打 warn，不阻断启动）
            voices.update(pickle.loads(p.read_bytes()))
            print(f"[startup] loaded {len(voices)} cached voices", flush=True)
        except Exception as e:
            print(f"[warn] 音色缓存加载失败: {e}", flush=True)


def _save_voices():
    # 先在锁内拷一份快照，再在锁外写盘，避免长时间持锁做 IO
    with voice_lock:
        data = dict(voices)
    # ⚠️ 写盘在锁外，理论上并发注册可能丢更新（当前 QPS 无影响）
    # ⚠️ 没有原子写（无 tmp+rename），写一半崩溃会留下损坏的 store.pkl
    _voices_store_path().write_bytes(pickle.dumps(data))


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


# ---------- 同步接口 ----------

class SpeakRequest(BaseModel):
    text: str
    speaker: str
    language: str | None = None
    instruct: str | None = None


class CloneRequest(BaseModel):
    text: str
    voice_id: str
    language: str | None = None


@app.post("/v1/speak", dependencies=[Depends(require_api_key)])
def speak(req: SpeakRequest):
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
        #   text              要合成的文本（可传 list 批量；本服务每次只传一条）
        #   speaker           canonical_speaker 归一后的官方音色名
        #   language          语种名，None = 自动检测
        #   instruct          语气/风格指令，如「用开心的语气」
        #                     ⚠️⚠️ 对本项目完全无效 ⚠️⚠️
        #                     源码 qwen3_tts_model.py:799：
        #                         if self.model.tts_model_size in "0b6":
        #                             instruct = None   # for 0b6 model, instruct is not supported
        #                     本服务用的是 Qwen3-TTS-12Hz-0.6B-CustomVoice（0.6B），
        #                     该模型不支持 instruct，参数会被直接置 None 丢弃。
        #                     只有 1.7B 的 CustomVoice 才支持。README 的示例
        #                     "instruct":"用开心的语气" 实际不产生任何效果。
        #   non_streaming_mode ⚠️ 不是「流式输出」开关
        #                     源码 docstring 明确：该参数为 false 时也只是「模拟流式
        #                     文本输入」，并不开启真正的流式输入或流式生成。
        #                     无论 true/false，返回值都是完整的 List[np.ndarray]。
        #   **kwargs          还可直通 HuggingFace generate() 的采样参数，例如
        #                     do_sample / top_k / top_p / temperature /
        #                     repetition_penalty / max_new_tokens
        #                     —— 本服务没有暴露它们，用的是模型 generate_config.json 里的默认值
        #
        # 返回值 Tuple[List[np.ndarray], int] = (wavs, sr)：
        #   wavs : list[np.ndarray]，float32，取值范围 [-1.0, 1.0]，长度=len(text)
        #   sr   : 采样率（int），随模型一起固定
        # 内部带 @torch.no_grad()，不会构建反向图
        wavs, sr = speak_model.generate_custom_voice(
            text=req.text, speaker=speaker,
            language=req.language, instruct=req.instruct,
            non_streaming_mode=True)
    # 本服务一次只合成一条 → 取 wavs[0]；转 16bit PCM WAV 返回
    return wav_response(wavs[0], sr)


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
        #   text              要合成的文本
        #   language          语种，None = 自动检测
        #   voice_clone_prompt  两种形态都接受：
        #                       Union[Dict[str, Any], List[VoiceClonePromptItem]]
        #                       本服务传的是后者（create_voice_clone_prompt 的返回值）
        #   non_streaming_mode  默认 False；本服务显式传 True
        #                       ⚠️ 同样不是流式输出开关，见 speak 里的说明
        #   **kwargs            同样可直通 HF generate() 采样参数（本服务未暴露）
        #
        # 返回值同 generate_custom_voice：(wavs: List[np.ndarray], sr: int)
        #   ⚠️ 克隆链路比自带音色慢得多（实测 p50 204s vs 78s），
        #      因为 Base 模型要额外处理 ref_code + 说话人向量
        wavs, sr = clone_model.generate_voice_clone(
            text=req.text, language=req.language,
            voice_clone_prompt=prompt, non_streaming_mode=True)
    return wav_response(wavs[0], sr)


@app.post("/v1/clone/upload", dependencies=[Depends(require_api_key)])
async def clone_upload(
        text: str = Form(...),
        ref_audio: UploadFile = File(...),
        ref_text: str = Form(None),
        language: str = Form(None)):
    # 模型只认本地 wav 路径 / URL / base64 / (ndarray, sr)，不认 UploadFile 对象
    # → 先把上传内容落盘成临时 wav，用完删掉
    tmp = TASKS_DIR / f"upload_{uuid.uuid4().hex}.wav"
    tmp.write_bytes(await ref_audio.read())
    try:
        # ⚠️ 本端点是 async def，但下面的模型调用是同步阻塞的
        #    → 推理期间会卡住整个事件循环，连 /health 都无法响应
        #    （对比：speak / clone 是 def，跑在线程池，不卡事件循环）
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
            # ⚠️⚠️ ref_text 实际是必填的 ⚠️⚠️
            #   x_vector_only_mode 默认 False → 走 ICL 模式 → 源码强制要求 ref_text：
            #       qwen3_tts_model.py:435
            #           if not xvec_only:
            #               if rtext is None or rtext == "":
            #                   raise ValueError("ref_text is required when
            #                       x_vector_only_mode=False (ICL mode). Bad index=...")
            #   而本端点的 ref_text 是 Form(None) 可选参数 ——
            #   ⚠️ 调用方不传 ref_text 会直接抛 ValueError → 500，README 却标成 ref_text? 可选
            #   两个解法（当前代码都没做）：
            #     a) 把 ref_text 改成 Form(...) 必填
            #     b) 显式传 x_vector_only_mode=True，退化成「只用说话人向量」克隆，
            #        此时 ref_text 可省略（但克隆相似度会下降）
            #
            # x_vector_only_mode=True  → 只用说话人向量（ref_spk_embedding），
            #                            忽略 ref_text / ref_code
            # x_vector_only_mode=False → ICL 模式，用 ref_code + ref_text 做上下文
            #
            # 返回 List[VoiceClonePromptItem]，元素字段（qwen3_tts_model.py:41）：
            #     ref_code           Optional[torch.Tensor]  # (T, Q) 或 (T,)
            #     ref_spk_embedding  torch.Tensor            # (D,)
            #     x_vector_only_mode bool
            #     icl_mode           bool
            #     ref_text           Optional[str]
            # 内部做的事：speech_tokenizer.encode(参考音频) 抽 ref_code；
            #             extract_speaker_embedding(重采样到 24k 的音频) 抽说话人向量
            prompt = clone_model.create_voice_clone_prompt(
                str(tmp), ref_text=ref_text or None)
            # ============ 模型调用 2/2：合成 ============
            # voice_clone_prompt 直接喂上面现算的 prompt，参数含义见 /v1/clone
            wavs, sr = clone_model.generate_voice_clone(
                text=text, language=language,
                voice_clone_prompt=prompt, non_streaming_mode=True)
    finally:
        # ⚠️ 进程在这里崩溃 / 被 kill 时 finally 不执行 → 会留下 upload_*.wav 孤儿文件
        tmp.unlink(missing_ok=True)
    return wav_response(wavs[0], sr)


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
        # ⚠️ 返回值是小写，所以前端拿到的音色名与官方文档大小写可能不一致
        "speakers": speak_model.get_supported_speakers(),
        "languages": speak_model.get_supported_languages(),
        "clone_voices": registered,
    }


@app.post("/v1/voices", dependencies=[Depends(require_api_key)])
async def register_voice(
        ref_audio: UploadFile = File(...),
        ref_text: str = Form(None),
        name: str = Form("")):
    # 同 clone_upload：模型只认路径，先落盘再删
    tmp = TASKS_DIR / f"upload_{uuid.uuid4().hex}.wav"
    tmp.write_bytes(await ref_audio.read())
    try:
        # ⚠️ async def + 同步阻塞模型调用 → 同样会卡住事件循环（见 clone_upload）
        with infer_slots:
            # ============ 模型调用：抽取并固化音色特征 ============
            # create_voice_clone_prompt(ref_audio, ref_text=None, x_vector_only_mode=False)
            #   -> List[VoiceClonePromptItem]
            # 这是整条链路最贵的一步（音频编码 + 说话人向量提取），
            # 做完缓存进 voices，之后 /v1/clone 就能反复复用，不用重算
            #
            # ⚠️⚠️ ref_text 实际必填 ⚠️⚠️
            #   默认 x_vector_only_mode=False（ICL 模式），源码 qwen3_tts_model.py:435 会
            #   对 None / 空字符串直接 raise ValueError：
            #       "ref_text is required when x_vector_only_mode=False (ICL mode)"
            #   而本接口把 ref_text 声明成 Form(None) 可选 ——
            #   ⚠️ 不传 ref_text 就 500。README 写的 ref_text? 可选与实际行为不符。
            #   想让它可选，必须显式传 x_vector_only_mode=True（仅用说话人向量）。
            prompt = clone_model.create_voice_clone_prompt(
                str(tmp), ref_text=ref_text or None)
    finally:
        tmp.unlink(missing_ok=True)
    voice_id = uuid.uuid4().hex[:12]
    # entry 结构 = voices[voice_id] 的值，也是 store.pkl 里存的东西
    entry = {
        # 未指定 name 时用 voice_id 前 6 位自动生成
        "name": name or f"voice-{voice_id[:6]}",
        # ★ 核心字段：List[VoiceClonePromptItem]，内含 torch.Tensor
        #   直接被 /v1/clone 与 _run_task 的 clone 分支消费
        #   ⚠️ 这就是 store.pkl 与 qwen_tts / torch 版本绑定的原因
        "prompt": prompt,
        "ref_text": ref_text,
        "created_at": time.time(),
    }
    with voice_lock:
        voices[voice_id] = entry
    # 立刻落盘，保证重启不丢。⚠️ 音色只增不减，无删除接口，store.pkl 只会变大
    _save_voices()
    return {"voice_id": voice_id, "name": entry["name"]}


# ---------- 异步任务 ----------

class TaskRequest(BaseModel):
    type: str                      # "speak" | "clone"
    text: str
    speaker: str | None = None
    voice_id: str | None = None
    language: str | None = None
    instruct: str | None = None


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
    # ⚠️ 惰性清理：本函数只在 POST /v1/tasks 里被调用，没有后台定时任务
    #    长期没有新任务提交 → 旧 wav 永远不会被回收


def _run_task(task_id: str, req: TaskRequest):
    # 在 executor 线程里执行，先标记 running
    with tasks_lock:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["started_at"] = time.time()
    try:
        # 同样走 infer_slots 限流；executor 的 max_workers 与它同为 MAX_CONCURRENT
        with infer_slots:
            if req.type == "speak":
                # ---- 模型调用：自带音色（参数含义见 /v1/speak）----
                # ⚠️ 这里的 instruct 同样会被 0.6B 模型丢弃（qwen3_tts_model.py:799）
                # ⚠️ canonical_speaker 在这里才校验，但 create_task 已提前校验过一次，
                #    正常不会走到异常分支
                wavs, sr = speak_model.generate_custom_voice(
                    text=req.text, speaker=canonical_speaker(req.speaker),
                    language=req.language, instruct=req.instruct,
                    non_streaming_mode=True)
            else:
                # ---- 模型调用：克隆音色（参数含义见 /v1/clone）----
                # prompt 由注册时算好并缓存，这里直接复用，不再付特征提取的开销
                wavs, sr = clone_model.generate_voice_clone(
                    text=req.text, language=req.language,
                    voice_clone_prompt=get_voice_prompt(req.voice_id),
                    non_streaming_mode=True)
        path = TASKS_DIR / f"{task_id}.wav"
        # 复用 wav_response 只为拿 WAV 字节：它内部已把 float32 量化成 16bit PCM
        # ⚠️ 这里 new 出来的 Response 的 X-Audio-Duration-Sec 头被丢弃了
        #    （duration 另存到 duration_sec 字段，所以功能上没损失）
        path.write_bytes(wav_response(wavs[0], sr).body)
        with tasks_lock:
            tasks[task_id].update(
                status="succeeded",
                finished_at=time.time(),
                # 音频时长（秒）= 样本数 / 采样率
                duration_sec=round(len(wavs[0]) / sr, 2))
    except Exception as e:
        # 任何异常（含模型 ValueError、voice_id 失效）都只记到内存状态里
        # ⚠️ 不重试、不告警；且状态在内存中，服务一重启就没了
        with tasks_lock:
            tasks[task_id].update(status="failed", error=str(e))


@app.post("/v1/tasks", dependencies=[Depends(require_api_key)])
def create_task(req: TaskRequest):
    # 顺带触发一次惰性清理（这是唯一的清理入口）
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
    # ⚠️ type=clone 时 req.instruct 会被静默忽略（只有 speak 分支用得到），
    #    而且 speak 分支的 instruct 对 0.6B 也无效 —— 这个字段实际是双重无效
    task_id = uuid.uuid4().hex
    with tasks_lock:
        tasks[task_id] = {
            "task_id": task_id, "type": req.type, "status": "queued",
            # 只留前 80 字，避免长文本把内存表撑大
            "text": req.text[:80], "created_at": time.time()}
    # 提交即返回（秒回），实际合成在 executor 线程里跑
    # ⚠️ executor 队列无界：高并发提交时大量任务会停在 queued，没有背压/拒绝机制
    executor.submit(_run_task, task_id, req)
    return {"task_id": task_id, "status": "queued",
            "poll": f"/v1/tasks/{task_id}"}


@app.get("/v1/tasks/{task_id}", dependencies=[Depends(require_api_key)])
def task_status(task_id: str):
    with tasks_lock:
        t = tasks.get(task_id)
    # ⚠️ 任务状态在内存里：服务重启后一律 404，即使对应的 wav 还在磁盘上（孤儿文件）
    if t is None:
        raise HTTPException(404, f"未知任务: {task_id}")
    # ⚠️ 死代码：entry 里从来没有写过 audio_path 字段，这个过滤无实际效果
    out = {k: v for k, v in t.items() if k != "audio_path"}
    if t["status"] == "succeeded":
        out["audio_url"] = f"/v1/tasks/{task_id}/audio"
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
    # ⚠️ 免鉴权，且不检查 API_KEYS 是否配置 —— key 没配时这里照样 200，
    #    但除本接口外的全部接口都会 500。做存活探测时要注意这个盲区。
    return {
        "status": "ok",
        # 两个模型是否都加载成功（lifespan 跑完才为 True）
        "models_loaded": speak_model is not None and clone_model is not None,
        "voices": len(voices),
        "tasks": len(tasks),
    }
