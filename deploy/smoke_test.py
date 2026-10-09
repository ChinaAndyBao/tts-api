"""Qwen3-TTS 冒烟测试：自带音色合成 + 声音克隆（CPU）。"""
import os
import time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = "/root/tts-out"
MODELS = "/root/models"
os.makedirs(OUT, exist_ok=True)


def load(name):
    t0 = time.time()
    m = Qwen3TTSModel.from_pretrained(
        os.path.join(MODELS, name),
        device_map="cpu",
        dtype=torch.float32,
    )
    print(f"[load] {name}: {time.time() - t0:.1f}s")
    return m


cv = load("Qwen3-TTS-12Hz-0.6B-CustomVoice")
print("[info] speakers:", cv.get_supported_speakers())
print("[info] languages:", cv.get_supported_languages())

REF_TEXT = "你好，这是一段用于验证安装的测试语音。"
t0 = time.time()
wavs, sr = cv.generate_custom_voice(
    text=REF_TEXT,
    language="Chinese",
    speaker="Vivian",
    instruct="用平静的语气说",
)
sf.write(f"{OUT}/custom_voice.wav", wavs[0], sr)
print(f"[gen] custom_voice.wav: {time.time() - t0:.1f}s, sr={sr}, dur={len(wavs[0]) / sr:.2f}s")

base = load("Qwen3-TTS-12Hz-0.6B-Base")
t0 = time.time()
wavs2, sr2 = base.generate_voice_clone(
    text="现在听到的是声音克隆生成的第二段测试语音。",
    language="Chinese",
    ref_audio=f"{OUT}/custom_voice.wav",
    ref_text=REF_TEXT,
)
sf.write(f"{OUT}/voice_clone.wav", wavs2[0], sr2)
print(f"[gen] voice_clone.wav: {time.time() - t0:.1f}s, sr={sr2}, dur={len(wavs2[0]) / sr2:.2f}s")

for f in ("custom_voice.wav", "voice_clone.wav"):
    size = os.path.getsize(f"{OUT}/{f}")
    assert size > 32000, f"{f} too small ({size}B)"
    print(f"[check] {f}: {size / 1024:.0f}KB ok")
print("SMOKE TEST OK")
