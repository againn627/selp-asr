"""下载模型文件。

选源策略（三条规则，按顺序生效）：

1. **看出口 IP 定默认顺序**
   查一次公网出口 IP 的归属地（走代理时查到的就是代理出口，所以开了代理就自动按代理区域算）：
   - 中国大陆 → 先试 hf-mirror.com（国内镜像）
   - 其他地区 → 先试 huggingface.co（官方）
   查不到就按中国大陆处理（本项目主要用户在国内，且镜像对国外也能通）。

2. **实测覆盖地区判断**
   按默认顺序对源做一次 2 MB 抽样测速，明显更快的排前面。
   地区判断只是「没有测速数据时的猜法」，测出来谁快就用谁。

3. **下载中慢就换**
   开始下载后若连续 10 秒速度低于 100 KB/s，换下一个源；.part 文件保留，换源接着下。
   全部源都失败时打印手动下载地址。

每个文件带精确字节数（2026-11-01 实测），下完对不上就重下，避免留下半个文件。

用法：
    python -m asr_server.download            下载到默认目录
    python -m asr_server.download --check    只检查完整性
    python -m asr_server.download --source   只打印选源结果（不下载）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_REPO = "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
VAD_GITHUB = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

# 两个 HF 源，内容是同一份文件，可以互相续传
SOURCE_MIRROR = "https://hf-mirror.com"
SOURCE_HF = "https://huggingface.co"

# 必需文件 + 精确字节数
REQUIRED_FILES: dict[str, int] = {
    "conv_frontend.onnx": 44_148_281,
    "encoder.int8.onnx": 182_491_662,
    "decoder.int8.onnx": 755_914_231,
    "tokenizer/vocab.json": 2_776_833,
    "tokenizer/merges.txt": 1_671_853,
    "tokenizer/tokenizer_config.json": 12_487,
}

# 示例音频：能下就下（自检用），下不下来不影响识别
OPTIONAL_FILES: dict[str, int] = {
    "test_wavs/raokouling.wav": 1_831_074,
    "test_wavs/fast1.wav": 1_003_794,
    "test_wavs/noise2.wav": 741_186,
    "test_wavs/cantonese.wav": 526_444,
    "test_wavs/qiqiu1.wav": 1_631_150,
}

VAD_BYTES = 643_854
TOTAL_BYTES = sum(REQUIRED_FILES.values())

SLOW_BYTES_PER_SEC = 100 * 1024   # 低于这个速度算慢
SLOW_WINDOW_SECONDS = 10          # 连续这么久才算慢，避免瞬时抖动就换源
PROBE_BYTES = 4 * 1024 * 1024     # 测速抽样大小（小文件会被 TLS 握手时间主导）
PROBE_SECONDS = 8                 # 测速最多等这么久

CHINA = "中国"          # 哨兵值，区别于 ISO 国家码


def _human(n: float) -> str:
    if n < 1024:
        return f"{n:.0f} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
    return f"{n:.1f} GB"


def _bar(done: int, total: int, width: int = 22) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = max(0, min(width, int(width * done / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------- 选源 ----------------

def detect_region(timeout: float = 5.0) -> str:
    """查公网出口 IP 的归属地。走代理时就是代理出口的地区。

    返回 "中国" 或 ISO 国家码（如 "HK"、"US"）；查不到返回 "中国"。
    """
    try:
        req = urllib.request.Request(
            "https://www.cloudflare.com/cdn-cgi/trace",
            headers={"User-Agent": "selp-asr/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", "replace")
        fields = dict(
            line.split("=", 1) for line in body.splitlines() if "=" in line
        )
        loc = (fields.get("loc") or "").strip().upper()
        ip = fields.get("ip", "?")
        if loc == "CN":
            return CHINA
        if loc:
            print(f"    出口 IP {ip}，地区 {loc} → 使用官方源优先")
            return loc
    except Exception:
        pass

    # 兜底：另一个只回国家的接口
    try:
        with urllib.request.urlopen(
            "http://ip-api.com/json/?fields=countryCode,country", timeout=timeout
        ) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        code = (data.get("countryCode") or "").strip().upper()
        if code == "CN":
            return CHINA
        if code:
            print(f"    地区 {data.get('country')}（{code}）→ 使用官方源优先")
            return code
    except Exception:
        pass

    print("    查不到出口地区，按中国大陆处理（镜像优先）")
    return CHINA


def probe_speed(url: str, seconds: float = PROBE_SECONDS, size: int = PROBE_BYTES) -> float:
    """抽样下载，返回字节/秒；完全拿不到数据返回 0。

    关键：只有「一个字节都拿不到」才算连不上。慢但能拿到数据就算可用，
    给一个很低的速度值而不是 0——否则会把能用的源误判成不可用。
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "selp-asr/1.0", "Range": f"bytes=0-{size - 1}"},
    )
    started = time.time()
    got = 0
    try:
        with urllib.request.urlopen(req, timeout=seconds) as resp:
            while time.time() - started < seconds:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                got += len(chunk)
                if got >= size:
                    break
    except Exception:
        pass
    elapsed = max(time.time() - started, 0.001)
    if got == 0:
        return 0.0
    # 极少数据时按实际耗时算会把速度算得虚高，用下限兜住
    return got / max(elapsed, 0.5)


def ordered_bases(probe: bool = True) -> list[str]:
    """返回按优先级排好的源列表。"""
    region = detect_region()
    if region == CHINA:
        order = [SOURCE_MIRROR, SOURCE_HF]
        print("    国内网络 → 国内镜像优先")
    else:
        order = [SOURCE_HF, SOURCE_MIRROR]
        print("    海外网络 → 官方源优先")

    if not probe:
        return order

    # 实测覆盖：拿真实的大文件（encoder）测，避免小文件被 TLS 握手时间主导
    probe_path = "encoder.int8.onnx"
    speeds: dict[str, float] = {}
    for base in order:
        url = f"{base}/{MODEL_REPO}/resolve/main/{probe_path}"
        speed = probe_speed(url, seconds=PROBE_SECONDS, size=PROBE_BYTES)
        speeds[base] = speed
        label = base.split("/")[2]
        if speed <= 0:
            print(f"    测速 {label}：连不上")
        else:
            print(f"    测速 {label}：{_human(speed)}/s")

    fastest = max(speeds, key=lambda k: speeds[k])
    slowest = min(speeds, key=lambda k: speeds[k])
    if speeds[fastest] > 0 and fastest != order[0] and speeds[fastest] > speeds[slowest] * 2:
        order = [fastest, slowest]
        print(f"    实测 {fastest.split('/')[2]} 更快 → 改用它")
    elif speeds[order[0]] == 0 and speeds[order[1]] > 0:
        order = [order[1], order[0]]
        print(f"    首选连不上 → 换用 {order[0].split('/')[2]}")

    return order


# ---------------- 下载 ----------------

def _download_from(url: str, dest: Path, expected: int | None) -> None:
    """从单个 URL 下载到 dest（支持 .part 续传）。慢或出错都抛异常。"""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if expected and have > expected:
        part.unlink()
        have = 0

    headers = {"User-Agent": "selp-asr/1.0"}
    if have and expected and have < expected:
        headers["Range"] = f"bytes={have}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        resuming = have > 0 and resp.status == 206
        if not resuming:
            have = 0
        total = expected or 0
        started = time.time()
        window_start = started
        window_bytes = 0
        last_print = 0.0
        with part.open("ab" if resuming else "wb") as fh:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                have += len(chunk)
                window_bytes += len(chunk)
                now = time.time()

                # 慢就放弃这个源，让调用方换一个
                if now - window_start >= SLOW_WINDOW_SECONDS:
                    rate = window_bytes / (now - window_start)
                    if rate < SLOW_BYTES_PER_SEC:
                        raise OSError(
                            f"速度太慢（{_human(rate)}/s），换一个源"
                        )
                    window_start, window_bytes = now, 0

                if now - last_print >= 0.2:
                    speed = have / max(now - started, 0.001)
                    pct = f" {have * 100 // total:3d}%" if total else ""
                    sys.stdout.write(
                        f"\r      {_bar(have, total)} {_human(have)}"
                        f"{('/' + _human(total)) if total else ''}{pct}"
                        f"  {_human(speed)}/s   "
                    )
                    sys.stdout.flush()
                    last_print = now
    sys.stdout.write("\r" + " " * 72 + "\r")

    size = part.stat().st_size
    if expected and size != expected:
        raise OSError(f"大小不符：期望 {expected} 字节，实际 {size} 字节")
    part.replace(dest)


class FetchError(Exception):
    """所有源都失败。"""


def fetch_file(
    dest: Path, expected: int | None, bases: list[str], rel_path: str, label: str
) -> bool:
    """按源顺序下载一个文件。成功返回 True。"""
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        return True
    if dest.exists() and expected:
        dest.unlink()

    for idx, base in enumerate(bases, 1):
        url = f"{base}/{MODEL_REPO}/resolve/main/{rel_path}"
        host = base.split("/")[2]
        print(f"    {label}  源 {idx}/{len(bases)}：{host}")
        for attempt in range(1, 3):
            try:
                _download_from(url, dest, expected)
                return True
            except (urllib.error.URLError, OSError) as exc:
                part = dest.with_name(dest.name + ".part")
                kept = part.stat().st_size if part.exists() else 0
                print(f"      失败：{exc}")
                if kept:
                    print(f"      已收 {_human(kept)}，换源后接着下")
                if attempt == 1:
                    time.sleep(1)
    return False


# ---------------- 校验与编排 ----------------

def verify_model_dir(path: Path) -> list[str]:
    """返回缺失/损坏的文件描述；空列表表示完整。"""
    problems: list[str] = []
    for name, size in REQUIRED_FILES.items():
        f = path / name
        if not f.exists():
            problems.append(f"缺少 {name}")
        elif f.stat().st_size != size:
            problems.append(f"{name} 大小不对（{f.stat().st_size} != {size}）")
    return problems


def manual_hint(model_dir: Path) -> None:
    print()
    print("  自动下载失败了。可以手动下载后放到这个目录：")
    print(f"    {model_dir}")
    print("  需要的文件（保持下面的相对路径）：")
    for name, size in REQUIRED_FILES.items():
        print(f"    {name:32s} {_human(size)}")
    print()
    print("  下载地址二选一（把 {文件} 换成上面列的文件名）：")
    print(f"    {SOURCE_MIRROR}/{MODEL_REPO}/resolve/main/{{文件}}")
    print(f"    {SOURCE_HF}/{MODEL_REPO}/resolve/main/{{文件}}")
    print()


def ensure_model(model_dir: Path, quiet: bool = False) -> bool:
    """确保模型完整，缺什么下什么。返回是否成功。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    if not verify_model_dir(model_dir):
        if not quiet:
            print(f"  模型已就绪：{model_dir}")
            print(f"  共 {_human(TOTAL_BYTES)}")
        return True

    print("  需要下载模型，共约 " + _human(TOTAL_BYTES))
    print("  第一次装要等几分钟，以后不用再下。")
    print()
    bases = ordered_bases()
    print()

    ok = True
    for index, (name, size) in enumerate(REQUIRED_FILES.items(), 1):
        sys.stdout.write(f"  [{index}/{len(REQUIRED_FILES)}] {name}（{_human(size)}）\n")
        sys.stdout.flush()
        if not fetch_file(model_dir / name, size, bases, name, name):
            ok = False

    if not ok:
        manual_hint(model_dir)
        return False

    (model_dir / ".manifest.json").write_text(
        json.dumps(
            {"repo": MODEL_REPO, "bytes": TOTAL_BYTES, "files": REQUIRED_FILES},
            indent=2,
        ),
        encoding="utf-8",
    )

    # 示例音频：自检用，失败不阻塞
    for name, size in OPTIONAL_FILES.items():
        if not fetch_file(model_dir / name, size, bases, name, name):
            print("    （示例音频没下成，不影响识别）")

    problems = verify_model_dir(model_dir)
    if problems:
        for p in problems:
            print("  校验未通过：", p)
        return False
    return True


def ensure_vad(vad_path: Path) -> bool:
    """VAD 只用于长音频分段，缺失不影响短句，所以失败也不算错。"""
    if vad_path.exists() and vad_path.stat().st_size == VAD_BYTES:
        return True
    print(f"  （可选）长音频分段模型 {_human(VAD_BYTES)}")
    vad_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _download_from(VAD_GITHUB, vad_path, VAD_BYTES)
        return True
    except Exception as exc:
        print(f"      跳过（{exc}）")
        return False


def main() -> int:
    from asr_server.paths import model_dir, vad_model_path

    parser = argparse.ArgumentParser(description="下载 selp-asr 模型文件")
    parser.add_argument("--check", action="store_true", help="只检查完整性")
    parser.add_argument("--source", action="store_true", help="只测速选源，不下载")
    parser.add_argument("--dir", default="", help="模型目录")
    args = parser.parse_args()

    target = Path(args.dir) if args.dir else model_dir()

    if args.check:
        problems = verify_model_dir(target)
        if problems:
            print("模型不完整：")
            for p in problems:
                print("  -", p)
            return 1
        print(f"模型完整：{target}（{_human(TOTAL_BYTES)}）")
        return 0

    if args.source:
        print("选源测试：")
        for base in ordered_bases():
            print("  ", base)
        return 0

    ok = ensure_model(target)
    if ok:
        ensure_vad(vad_model_path())
        print()
        print(f"  完成，模型位置：{target}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
