"""线程数与解码耗时对比：python -m scripts.bench

同一批官方测试音频，跑不同线程数，看 RTF。
换机器后建议重跑一次确定 config.toml 的 num_threads。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asr_server.audio import decode_audio  # noqa: E402
from asr_server.config import load_config  # noqa: E402
from asr_server.engine import AsrEngine  # noqa: E402
from asr_server.paths import model_dir, vad_model_path  # noqa: E402

WAV_DIR = model_dir() / "test_wavs"
CLIPS = ["noise2.wav", "fast1.wav", "raokouling.wav"]
THREADS = (2, 4, 6, 8, 12)


def main() -> int:
    cfg = load_config()
    clips = []
    for name in CLIPS:
        path = WAV_DIR / name
        if path.exists():
            samples, sr = decode_audio(path.read_bytes())
            clips.append((name, samples, sr))

    if not clips:
        print(f"找不到测试音频，应该在这里：{WAV_DIR}")
        print("先跑 python -m asr_server.download 下载模型（自带 test_wavs）")
        return 1

    for threads in THREADS:
        engine = AsrEngine(
            model_dir=model_dir(),
            num_threads=threads,
            max_new_tokens=cfg.max_new_tokens,
            max_total_len=cfg.max_total_len,
            hotwords=cfg.hotwords,
            idle_unload_seconds=0,
            vad_model=vad_model_path(),
            vad_max_speech_seconds=cfg.vad_max_speech_seconds,
        )
        load_cost = engine.warmup()
        line = [f"threads={threads:2d} load={load_cost:.2f}s"]
        for name, samples, sr in clips:
            engine.transcribe(samples, sr)  # 预热一次，避开首句抖动
            started = time.perf_counter()
            res = engine.transcribe(samples, sr)
            cost = time.perf_counter() - started
            line.append(f"{name} {res.duration:.1f}s->{cost:.2f}s RTF {cost / res.duration:.3f}")
        print(" | ".join(line), flush=True)
        engine.unload("bench")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
