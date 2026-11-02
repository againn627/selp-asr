"""麦克风测试：按住/回车说话，走 HTTP 让服务识别。

和旧版 record.py 的关键区别：不再自己加载模型，而是把录音发给已启动的服务。
理由：服务已占约 1.1 GB，进程内再来一份就是 2.2 GB；而且这样测的正是「服务能不能用」。

用法：
    python -m asr_server.mic                 回车开始，再回车停止，循环
    python -m asr_server.mic -s 5            录 5 秒自动停
    python -m asr_server.mic --devices       列出录音设备
    python -m asr_server.mic --url http://127.0.0.1:8765 --token local-asr
"""
from __future__ import annotations

import argparse
import io
import time

import numpy as np

SAMPLE_RATE = 16000


def _level_bar(peak: float) -> str:
    """把音量画成小条，一眼看出麦克风有没有在工作。"""
    n = min(20, int(peak * 100))
    return "[" + "#" * n + "-" * (20 - n) + "]"


def _merge(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(frames, axis=0)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return np.ascontiguousarray(audio, dtype=np.float32)


def _wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def record_until_enter(label: str = "说完按回车停止") -> np.ndarray:
    """回车开始，再按回车停止，全程显示音量。"""
    import sounddevice as sd

    frames: list[np.ndarray] = []

    def callback(indata, _frames, _time, _status):
        frames.append(indata.copy())
        peak = float(np.max(np.abs(indata))) if indata.size else 0.0
        print(f"\r  录音 {_level_bar(peak)} {peak:0.2f}  （{label}）", end="", flush=True)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback, blocksize=1024
    ):
        input()
    print("\r" + " " * 72 + "\r", end="", flush=True)
    return _merge(frames)


def record_fixed(seconds: float) -> np.ndarray:
    """录固定时长，全程显示音量。"""
    import sounddevice as sd

    frames: list[np.ndarray] = []

    def callback(indata, _frames, _time, _status):
        frames.append(indata.copy())
        peak = float(np.max(np.abs(indata))) if indata.size else 0.0
        print(
            f"\r  录音 {_level_bar(peak)} {peak:0.2f}  （开口说话，{seconds:g} 秒后停）",
            end="",
            flush=True,
        )

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback, blocksize=1024
    ):
        sd.sleep(int(seconds * 1000))
    print("\r" + " " * 72 + "\r", end="", flush=True)
    return _merge(frames)


def show_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def main() -> int:
    parser = argparse.ArgumentParser(description="麦克风录音 → 让本地服务识别")
    parser.add_argument("-s", "--seconds", type=float, default=0, help="录固定秒数后自动停")
    parser.add_argument("--devices", action="store_true", help="列出录音设备")
    parser.add_argument("--url", default="", help="服务地址，默认取配置")
    parser.add_argument("--token", default="", help="服务令牌，默认取配置")
    args = parser.parse_args()

    if args.devices:
        show_devices()
        return 0

    try:
        import sounddevice as sd
    except ImportError:
        print("缺少录音组件。安装：python -m pip install sounddevice")
        return 2

    from asr_server import client
    from asr_server.config import ConfigError, load_config

    # 服务地址：命令行 > 配置
    base_url = args.url
    token = args.token
    if not base_url:
        try:
            cfg = load_config()
        except ConfigError as exc:
            print(f"配置有问题：{exc}")
            return 1
        host = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
        base_url = f"http://{host}:{cfg.port}"
        token = token or cfg.token

    # 先确认服务在线，比录完再说「连不上」好
    try:
        info = client.health(base_url, token)
    except client.ClientError as exc:
        print()
        print("  连不上语音服务。")
        print(f"  地址：{base_url}")
        print(f"  原因：{exc}")
        print()
        print("  先启动服务：双击 start-server.bat")
        return 2

    print(f"  服务正常（模型{'已加载' if info.get('loaded') else '未加载'}）")

    try:
        mic = sd.query_devices(kind="input")
        print(f"  麦克风：{mic['name']}")
    except Exception as exc:
        print(f"  找不到麦克风：{exc}")
        print("  检查一下系统录音设备，或运行 --devices 看看列表。")
        return 2

    # 让服务在录音的同时把模型加载好，减少第一句的等待
    import threading

    threading.Thread(target=client.warmup, args=(base_url, token), daemon=True).start()

    def do_transcribe(audio: np.ndarray) -> None:
        if audio.size < SAMPLE_RATE // 10:
            print("  录音太短，跳过。")
            return
        peak = float(np.max(np.abs(audio)))
        if peak < 0.005:
            print(f"  几乎没收到声音（峰值 {peak:.4f}）。检查麦克风是否静音、是否选错设备。")
            return
        duration = audio.size / SAMPLE_RATE
        print("  识别中 ...", end="", flush=True)
        started = time.perf_counter()
        try:
            result = client.transcribe_bytes(
                base_url, _wav_bytes(audio), "mic.wav", token
            )
        except client.ClientError as exc:
            print(f"\r  识别失败：{exc}")
            return
        wall = time.perf_counter() - started
        print(f"\r  音频 {duration:.2f}s | 出字 {wall:.2f}s（服务端 {result.elapsed:.2f}s）")
        print(f"  >> {result.text or '（没识别到内容）'}")
        print()

    print()
    if args.seconds > 0:
        print(f"  准备录 {args.seconds:g} 秒，现在开始说话 ...")
        do_transcribe(record_fixed(args.seconds))
        return 0

    print("  回车开始说话，说完再按回车出字；输入 q 回车退出。")
    while True:
        print("  按回车开始（q 退出）：", end="", flush=True)
        try:
            reply = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if reply == "q":
            break
        do_transcribe(record_until_enter())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
