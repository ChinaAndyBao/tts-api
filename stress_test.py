#!/usr/bin/env python3
"""tts-api 压测：voices(轻) / speak(同步重) / tasks(异步) / clone(克隆) / mixed(混合并发)。

用法示例：
  python stress_test.py voices -c 50 -n 1000
  python stress_test.py speak -c 8 -n 24 --text 压测
  python stress_test.py tasks -c 8 -n 24
  python stress_test.py clone -c 4 -n 8 --voice-id <vid>
  python stress_test.py mixed --speak-n 12 --speak-c 8 --tasks-n 12 --voices-n 300 --voices-c 32
"""
import argparse
import asyncio
import json
import os
import threading
import time
from datetime import datetime

import httpx

KEY_FILE = "/etc/tts-api.env"


def load_key():
    for line in open(KEY_FILE):
        if line.startswith("TTS_API_KEYS="):
            return line.strip().split("=", 1)[1].split(",")[0]
    raise SystemExit("TTS_API_KEYS not found in " + KEY_FILE)


def pctl(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


class Group:
    def __init__(self, name):
        self.name = name
        self.lat = []
        self.ok = 0
        self.fail = 0
        self.timeout = 0
        self.codes = {}

    def hit(self, dt, status, ok, timeout=False):
        if timeout:
            self.timeout += 1
        elif ok:
            self.ok += 1
            self.lat.append(dt)
        else:
            self.fail += 1
            self.codes[status] = self.codes.get(status, 0) + 1

    def report(self, wall):
        total = self.ok + self.fail + self.timeout
        rps = total / wall if wall else 0
        line = (f"[{self.name}] sent={total} ok={self.ok} non2xx={self.fail} "
                f"timeout={self.timeout} wall={wall:.1f}s rps={rps:.1f}")
        if self.lat:
            line += (f" | lat s: min={min(self.lat):.2f} p50={pctl(self.lat, 50):.2f} "
                     f"p90={pctl(self.lat, 90):.2f} p99={pctl(self.lat, 99):.2f} "
                     f"max={max(self.lat):.2f}")
        if self.codes:
            line += f" | codes={self.codes}"
        return line


class SysSampler(threading.Thread):
    def __init__(self, pid, interval=2.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.stop_evt = threading.Event()
        self.max_rss = 0.0
        self.samples = []

    def _cpu_ticks(self):
        with open(f"/proc/{self.pid}/stat") as f:
            parts = f.read().rsplit(") ", 1)[1].split()
        return int(parts[11]) + int(parts[12])

    def _rss_mb(self):
        with open(f"/proc/{self.pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
        return 0.0

    def run(self):
        clk = os.sysconf("SC_CLK_TCK")
        prev_cpu, prev_t = self._cpu_ticks(), time.time()
        while not self.stop_evt.wait(self.interval):
            try:
                cpu = self._cpu_ticks()
                now = time.time()
                pct = (cpu - prev_cpu) / clk / (now - prev_t) * 100
                rss = self._rss_mb()
                self.max_rss = max(self.max_rss, rss)
                self.samples.append((now, pct, rss))
                prev_cpu, prev_t = cpu, now
            except Exception:
                break

    def summary(self):
        if not self.samples:
            return "no samples"
        cpus = [s[1] for s in self.samples]
        return (f"cpu% avg={sum(cpus) / len(cpus):.0f} max={max(cpus):.0f} | "
                f"rss max={self.max_rss:.0f}MB")


async def one_speak(client, g, sem, text, speaker, instruct):
    async with sem:
        t0 = time.time()
        try:
            r = await client.post("/v1/speak", json={
                "text": text, "speaker": speaker,
                "language": "Chinese", "instruct": instruct})
            body_ok = r.status_code == 200 and len(r.content) > 8000
            g.hit(time.time() - t0, r.status_code, body_ok)
        except httpx.TimeoutException:
            g.hit(time.time() - t0, 0, False, timeout=True)
        except Exception:
            g.hit(time.time() - t0, 0, False)


async def one_voices(client, g, sem):
    async with sem:
        t0 = time.time()
        try:
            r = await client.get("/v1/voices")
            g.hit(time.time() - t0, r.status_code, r.status_code == 200)
        except httpx.TimeoutException:
            g.hit(time.time() - t0, 0, False, timeout=True)
        except Exception:
            g.hit(time.time() - t0, 0, False)


async def one_clone(client, g, sem, text, voice_id):
    async with sem:
        t0 = time.time()
        try:
            r = await client.post("/v1/clone", json={
                "text": text, "voice_id": voice_id, "language": "Chinese"})
            body_ok = r.status_code == 200 and len(r.content) > 8000
            g.hit(time.time() - t0, r.status_code, body_ok)
        except httpx.TimeoutException:
            g.hit(time.time() - t0, 0, False, timeout=True)
        except Exception:
            g.hit(time.time() - t0, 0, False)


async def one_task(client, g_sub, g_e2e, sem, text, speaker, deadline):
    t0 = time.time()
    async with sem:
        try:
            r = await client.post("/v1/tasks", json={
                "type": "speak", "text": text,
                "speaker": speaker, "language": "Chinese"})
            g_sub.hit(time.time() - t0, r.status_code, r.status_code == 200)
            if r.status_code != 200:
                return
            tid = r.json()["task_id"]
        except httpx.TimeoutException:
            g_sub.hit(time.time() - t0, 0, False, timeout=True)
            return
        except Exception:
            g_sub.hit(time.time() - t0, 0, False)
            return
    while time.time() - t0 < deadline:
        try:
            s = await client.get(f"/v1/tasks/{tid}")
            st = s.json().get("status")
            if st == "succeeded":
                a = await client.get(f"/v1/tasks/{tid}/audio")
                body_ok = a.status_code == 200 and len(a.content) > 8000
                g_e2e.hit(time.time() - t0, a.status_code, body_ok)
                return
            if st == "failed":
                g_e2e.hit(time.time() - t0, 500, False)
                return
        except Exception:
            pass
        await asyncio.sleep(2)
    g_e2e.hit(time.time() - t0, 0, False, timeout=True)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["voices", "speak", "tasks", "clone", "mixed"])
    p.add_argument("--url", default="http://127.0.0.1:9898")
    p.add_argument("--text", default="压测")
    p.add_argument("--speaker", default="vivian")
    p.add_argument("--voice-id", default="")
    p.add_argument("--deadline", type=float, default=540)
    p.add_argument("-c", "--concurrency", type=int, default=8)
    p.add_argument("-n", "--requests", type=int, default=24)
    p.add_argument("--speak-n", type=int, default=12)
    p.add_argument("--speak-c", type=int, default=8)
    p.add_argument("--tasks-n", type=int, default=12)
    p.add_argument("--voices-n", type=int, default=300)
    p.add_argument("--voices-c", type=int, default=32)
    p.add_argument("--clone-n", type=int, default=8)
    p.add_argument("--clone-c", type=int, default=4)
    return p.parse_args()


async def main():
    args = parse()
    key = load_key()
    groups = {}
    maxc = max(args.concurrency, args.speak_c, args.voices_c, args.clone_c, 8)

    pid = None
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-f", "uvicorn app:app"],
                             capture_output=True, text=True)
        pid = int(out.stdout.split()[0])
    except Exception:
        pass
    sampler = SysSampler(pid) if pid else None
    if sampler:
        sampler.start()

    timeout = httpx.Timeout(600.0, connect=10.0)
    limits = httpx.Limits(max_connections=maxc + 30,
                          max_keepalive_connections=maxc + 10)
    async with httpx.AsyncClient(base_url=args.url, timeout=timeout,
                                 limits=limits,
                                 headers={"X-API-Key": key}) as client:
        jobs = []
        t0 = time.time()

        if args.mode in ("voices", "mixed"):
            g = groups["voices"] = Group("voices")
            n = args.voices_n if args.mode == "mixed" else args.requests
            c = args.voices_c if args.mode == "mixed" else args.concurrency
            sem = asyncio.Semaphore(c)
            t = time.time()
            await asyncio.gather(*(one_voices(client, g, sem)
                                   for _ in range(n)))
            print(g.report(time.time() - t), flush=True)

        if args.mode in ("speak", "mixed"):
            g = groups["speak"] = Group("speak")
            n = args.speak_n if args.mode == "mixed" else args.requests
            c = args.speak_c if args.mode == "mixed" else args.concurrency
            sem = asyncio.Semaphore(c)
            t = time.time()
            await asyncio.gather(*(one_speak(client, g, sem, args.text,
                                             args.speaker, None)
                                   for _ in range(n)))
            print(g.report(time.time() - t), flush=True)

        if args.mode in ("clone", "mixed") and args.voice_id:
            g = groups["clone"] = Group("clone")
            n = args.clone_n if args.mode == "mixed" else args.requests
            c = args.clone_c if args.mode == "mixed" else args.concurrency
            sem = asyncio.Semaphore(c)
            t = time.time()
            await asyncio.gather(*(one_clone(client, g, sem, args.text,
                                             args.voice_id)
                                   for _ in range(n)))
            print(g.report(time.time() - t), flush=True)

        if args.mode == "tasks":
            g_sub = groups["tasks-submit"] = Group("tasks-submit")
            g_e2e = groups["tasks-e2e"] = Group("tasks-e2e")
            sem = asyncio.Semaphore(args.concurrency)
            t = time.time()
            await asyncio.gather(*(one_task(client, g_sub, g_e2e, sem,
                                            args.text, args.speaker,
                                            args.deadline)
                                   for _ in range(args.requests)))
            wall = time.time() - t
            print(g_sub.report(wall), flush=True)
            print(g_e2e.report(wall), flush=True)

        if args.mode == "mixed":
            g_sub = groups["tasks-submit"] = Group("tasks-submit")
            g_e2e = groups["tasks-e2e"] = Group("tasks-e2e")
            sem = asyncio.Semaphore(8)
            t = time.time()
            await asyncio.gather(*(one_task(client, g_sub, g_e2e, sem,
                                            args.text, args.speaker,
                                            args.deadline)
                                   for _ in range(args.tasks_n)))
            wall = time.time() - t
            print(g_sub.report(wall), flush=True)
            print(g_e2e.report(wall), flush=True)

        total_wall = time.time() - t0

    if sampler:
        sampler.stop_evt.set()
        sampler.join(timeout=5)
        print("[sys]", sampler.summary(), flush=True)

    result = {
        "mode": args.mode,
        "time": datetime.now().isoformat(),
        "total_wall": round(total_wall, 1),
        "groups": {name: {"ok": g.ok, "fail": g.fail,
                          "timeout": g.timeout, "codes": g.codes,
                          "p50": pctl(g.lat, 50), "p90": pctl(g.lat, 90),
                          "p99": pctl(g.lat, 99),
                          "min": min(g.lat) if g.lat else 0,
                          "max": max(g.lat) if g.lat else 0}
                   for name, g in groups.items()},
    }
    out = f"/root/tts-api/stress_{args.mode}_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved:", out, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
