"""命令行转录工具：python -m asr_server.cli <音频文件> [更多文件...]

不需要启动服务，直接进程内加载模型，适合快速验证效果。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asr_server.audio import decode_audio  # noqa: E402
from asr_server.config import load_config  # noqa: E402
from asr_server.engine import AsrEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="本地语音转文字（Qwen3-ASR-0.6B）")
    parser.add_argument("files", nargs="+", help="音频文件路径")
    parser.add_argument("--threads", type=int, default=0, help="覆盖配置的 CPU 线程数")
    args = parser.parse_args()

    cfg = load_config(ROOT / "config.toml")
    engine = AsrEngine(
        model_dir=ROOT / cfg.model_dir,
        num_threads=args.threads or cfg.num_threads,
        max_new_tokens=cfg.max_new_tokens,
        max_total_len=cfg.max_total_len,
        hotwords=cfg.hotwords,
        idle_unload_seconds=0,
        vad_model=ROOT / cfg.vad_model,
        vad_max_speech_seconds=cfg.vad_max_speech_seconds,
    )
    missing = engine.check_model()
    if missing:
        print("模型文件缺失：", "；".join(missing), file=sys.stderr)
        return 2

    for name in args.files:
        path = Path(name)
        if not path.exists():
            print(f"[跳过] 文件不存在：{path}", file=sys.stderr)
            continue
        samples, sr = decode_audio(path.read_bytes())
        started = time.perf_counter()
        result = engine.transcribe(samples, sr)
        total = time.perf_counter() - started
        rtf = total / result.duration if result.duration else 0
        print(f"\n===== {path.name} =====")
        print(f"音频 {result.duration:.2f}s | 总耗时 {total:.2f}s（含首次加载 {result.load_elapsed:.2f}s）| RTF {rtf:.2f}")
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
