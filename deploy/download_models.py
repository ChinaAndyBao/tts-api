#!/usr/bin/env python3
"""通过 ModelScope HTTP API 下载 Qwen3-TTS 模型仓库（带重试/续传）。"""
import json
import os
import time
import urllib.parse
import urllib.request

API = "https://modelscope.cn/api/v1/models/{mid}/repo"
REPOS = {
    "Qwen/Qwen3-TTS-Tokenizer-12Hz": "/root/models/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": "/root/models/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "/root/models/Qwen3-TTS-12Hz-0.6B-Base",
}


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_files(mid):
    data = json.loads(get(f"{API.format(mid=mid)}/files?Revision=master").decode())
    out = []
    for f in data.get("Data", {}).get("Files", []):
        p = f.get("Path") or ""
        if not p or p.endswith("/") or f.get("Type") in ("tree", "dir"):
            continue
        out.append((p, int(f.get("Size") or 0)))
    return out


def download(mid, path, dest, size):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest) and size and os.path.getsize(dest) == size:
        print(f"  skip (cached) {path}", flush=True)
        return
    url = f"{API.format(mid=mid)}?Revision=master&FilePath={urllib.parse.quote(path)}"
    for attempt in range(1, 7):
        try:
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            if size and have > size:
                have = 0
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=180) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            got = os.path.getsize(dest)
            if not size or got == size:
                print(f"  ok {path} ({got / 1e6:.1f}MB)", flush=True)
                return
            print(f"  size mismatch {path}: {got}/{size}; retrying", flush=True)
        except Exception as e:
            print(f"  attempt {attempt} for {path} failed: {e}", flush=True)
            time.sleep(2 * attempt)
    raise SystemExit(f"ERROR: failed to download {mid}/{path}")


for mid, dest_dir in REPOS.items():
    files = list_files(mid)
    total = sum(s for _, s in files)
    print(f"== {mid}: {len(files)} files, {total / 1e9:.2f} GB", flush=True)
    for path, size in files:
        download(mid, path, os.path.join(dest_dir, path), size)
print("ALL MODELS DOWNLOADED", flush=True)
