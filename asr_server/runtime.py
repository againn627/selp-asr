"""项目自带 Python 运行时：下载、校验、解压到 runtime/python。

为什么要有这个东西
------------------
以前的做法是在用户机器上找 Python，再建 .venv。问题是：

* Windows 上 PATH 里的 python 可能是某个软件自带的精简版（没有 venv 模块）；
* 只有 conda 的人，venv 会「依赖」conda —— conda 一卸载、一升级，项目直接跑不起来；
* 干脆没装 Python 的电脑，只能让用户先自己装一遍。

所以现在改成自带一份：下载官方 python-build-standalone（和 uv 用的是同一家），
解压到 ``runtime/python``，依赖直接装进这份私有运行时，不碰系统里任何 Python。
用户机器上有没有 Python、有没有 conda，都跟本项目无关。

版本信息全部来自项目根的 ``runtime.lock.json``（唯一定义处），
``scripts/install-runtime.ps1`` 也读同一个文件。

只用标准库，且兼容 Python 3.8+ —— 它要在「还没有运行时」的时候被跑起来。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = ROOT / "runtime.lock.json"
RUNTIME_DIR = ROOT / "runtime" / "python"
MARKER_FILE = RUNTIME_DIR / ".selp-asr-runtime.json"

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
PY_EXE = RUNTIME_DIR / ("python.exe" if IS_WIN else "bin/python3")

CHUNK = 256 * 1024
SLOW_BYTES_PER_SEC = 100 * 1024
SLOW_WINDOW_SECONDS = 10
PROBE_SECONDS = 8
PROBE_BYTES = 2 * 1024 * 1024


class RuntimeSetupError(Exception):
    """运行时准备失败，且能给出人能看懂的说明。"""


def _human(n: float) -> str:
    if n < 1024:
        return f"{n:.0f} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
    return f"{n:.1f} GB"


# ---------------- 版本与平台 ----------------

def lock() -> dict:
    """读 runtime.lock.json。文件缺失说明项目不完整。"""
    if not LOCK_FILE.exists():
        raise RuntimeSetupError(f"缺少 {LOCK_FILE.name}，请重新下载完整的项目文件")
    try:
        return json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeSetupError(f"{LOCK_FILE.name} 内容损坏：{exc}") from exc


def platform_key() -> str:
    """当前平台对应的 lock 条目名，如 windows-amd64。"""
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        arch = "amd64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        raise RuntimeSetupError(
            f"暂不支持这个 CPU 架构：{platform.machine()}。"
            "可以在别的机器上装好，把整个项目目录（含 runtime/）拷过来用。"
        )

    if IS_WIN:
        return f"windows-{arch}"
    if IS_MAC:
        return f"macos-{arch}"
    if sys.platform.startswith("linux"):
        return f"linux-{arch}"
    raise RuntimeSetupError(
        f"暂不支持这个系统：{sys.platform}。"
        "可以在 Windows / macOS / Linux 上装好，把整个项目目录（含 runtime/）拷过来用。"
    )


def asset() -> dict:
    """当前平台要下载的文件：asset / sha256 / size。"""
    data = lock()
    key = platform_key()
    entry = (data.get("platforms") or {}).get(key)
    if not entry:
        raise RuntimeSetupError(f"runtime.lock.json 里没有 {key} 的条目")
    return dict(entry, python=data.get("python", ""), release=data.get("release", ""))


def source_urls(entry: dict) -> list[str]:
    """按区域给出候选下载地址（境外 GitHub 优先，境内镜像优先）。"""
    data = lock()
    templates = data.get("sources") or {}
    release = entry.get("release") or data.get("release", "")
    asset_name = entry["asset"]

    gh = (templates.get("github") or "").format(release=release)
    mirror = (templates.get("nju") or "").format(release=release)
    if not gh or not mirror:
        raise RuntimeSetupError("runtime.lock.json 里的 sources 不完整")

    region = _region()
    order = [gh, mirror] if region != "CN" else [mirror, gh]
    return [f"{base}/{asset_name}" for base in order if base]


def _region() -> str:
    """出口 IP 所属地区；走代理时就是代理出口。查不到按国内处理。"""
    try:
        from asr_server.download import detect_region

        return "CN" if detect_region() == "中国" else "OTHER"
    except Exception:
        return "CN"


# ---------------- 状态 ----------------

def is_installed() -> bool:
    """运行时是否已就位，且和 lock 里的版本一致。"""
    if not PY_EXE.exists():
        return False
    try:
        marker = json.loads(MARKER_FILE.read_text(encoding="utf-8"))
        entry = asset()
    except Exception:
        return False
    return marker.get("sha256") == entry["sha256"]


def runtime_python() -> Path:
    """运行时的解释器路径。"""
    return PY_EXE


def require_python() -> Path:
    """要解释器但没装时，直接给出人能看懂的错误。"""
    if not PY_EXE.exists():
        raise RuntimeSetupError(
            f"项目自带的 Python 不在了（{PY_EXE}）。\n"
            "  重新双击 start-server.bat 会自动装回来。"
        )
    return PY_EXE


# ---------------- 下载 ----------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected: int) -> None:
    """下载到 dest，支持 .part 续传；太慢就抛异常让调用方换源。"""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > expected:
        part.unlink()
        have = 0
    if have == expected and expected > 0:
        part.replace(dest)
        return

    headers = {"User-Agent": "selp-asr/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        resuming = have > 0 and resp.status == 206
        if not resuming:
            have = 0
        started = time.time()
        window_start = started
        window_bytes = 0
        last_print = 0.0
        with part.open("ab" if resuming else "wb") as fh:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                have += len(chunk)
                window_bytes += len(chunk)
                now = time.time()

                if now - window_start >= SLOW_WINDOW_SECONDS:
                    rate = window_bytes / (now - window_start)
                    if rate < SLOW_BYTES_PER_SEC:
                        raise OSError(f"速度太慢（{_human(rate)}/s）")
                    window_start, window_bytes = now, 0

                if now - last_print >= 0.2:
                    speed = have / max(now - started, 0.001)
                    pct = have * 100 // expected if expected else 0
                    filled = int(24 * have / expected) if expected else 0
                    bar = "[" + "#" * filled + "-" * (24 - filled) + "]"
                    sys.stdout.write(
                        f"\r      {bar} {_human(have)}/{_human(expected)}"
                        f" {pct:3d}%  {_human(speed)}/s   "
                    )
                    sys.stdout.flush()
                    last_print = now
    sys.stdout.write("\r" + " " * 78 + "\r")

    size = part.stat().st_size
    if expected and size != expected:
        raise OSError(f"大小不符：期望 {expected} 字节，实际 {size} 字节")
    part.replace(dest)


def _probe(url: str) -> float:
    """抽样测速；拿不到任何数据返回 0。"""
    req = urllib.request.Request(
        url, headers={"User-Agent": "selp-asr/1.0", "Range": f"bytes=0-{PROBE_BYTES - 1}"}
    )
    started = time.time()
    got = 0
    try:
        with urllib.request.urlopen(req, timeout=PROBE_SECONDS) as resp:
            while time.time() - started < PROBE_SECONDS:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                got += len(chunk)
                if got >= PROBE_BYTES:
                    break
    except Exception:
        pass
    if got == 0:
        return 0.0
    return got / max(time.time() - started, 0.5)


def _ordered(urls: list[str], entry: dict) -> list[str]:
    """实测速度重排：谁快用谁，连不上的排后面。"""
    if len(urls) < 2:
        return urls
    print("  测速，选下载源 ...")
    speeds = []
    for url in urls:
        rate = _probe(url)
        host = url.split("/")[2]
        print(f"    {host:32s} {'连不上' if rate <= 0 else _human(rate) + '/s'}")
        speeds.append((rate, url))
    if all(rate <= 0 for rate, _ in speeds):
        return urls
    speeds.sort(key=lambda item: item[0], reverse=True)
    best = speeds[0][1].split("/")[2]
    print(f"    用 {best}")
    return [url for _, url in speeds]


def _fetch(entry: dict, into: Path) -> None:
    """把运行时压缩包下载到 into，多源重试。"""
    dest = into / entry["asset"]
    urls = _ordered(source_urls(entry), entry)
    errors: list[str] = []

    for index, url in enumerate(urls, 1):
        host = url.split("/")[2]
        print(f"  下载 Python {entry.get('python', '')}（{_human(entry['size'])}）")
        print(f"    源 {index}/{len(urls)}：{host}")
        for attempt in (1, 2):
            try:
                _download(url, dest, entry["size"])
                digest = _sha256(dest)
                if digest != entry["sha256"]:
                    dest.unlink(missing_ok=True)
                    raise OSError(f"校验失败（sha256 {digest[:12]}… 不符）")
                return
            except (urllib.error.URLError, OSError) as exc:
                errors.append(f"{host}: {exc}")
                print(f"      失败：{exc}")
                if attempt == 1:
                    time.sleep(1)

    data = lock()
    release = entry.get("release") or data.get("release", "")
    gh = (data.get("sources", {}).get("github") or "").format(release=release)
    mirror = (data.get("sources", {}).get("nju") or "").format(release=release)
    raise RuntimeSetupError(
        "Python 运行时下载失败，项目需要它才能运行：\n"
        + "\n".join(f"    - {line}" for line in errors)
        + "\n\n  可以试试：\n"
        "    1) 检查网络能不能打开 GitHub 或国内镜像\n"
        "    2) 手动下载后放进项目里的 runtime/_cache/ 目录，再双击 start-server.bat\n"
        f"       文件名必须是：{entry['asset']}\n"
        f"       GitHub：{gh}/{entry['asset']}\n"
        f"       镜像  ：{mirror}/{entry['asset']}\n"
    )


def _assert_runtime_not_in_use() -> None:
    """运行时正被占着（服务在跑）就报错，不去动它。

    为什么要单独查：Windows 允许给目录改名，哪怕里面的 python.exe 正被另一个
    进程跑着（试过 os.rename 同名重命名，不会报错，探不出来）。
    真正能探到的是「以读写方式打开 exe」：被运行时返回 Permission denied，
    没被用时能打开。用这个判定比看 rename 结果可靠。
    """
    if not PY_EXE.exists():
        return
    try:
        handle = os.open(PY_EXE, os.O_RDWR)
    except OSError as exc:
        raise RuntimeSetupError(
            f"项目自带的 Python 正在被使用（{exc.strerror or exc}），现在不能更换。\n"
            "  先双击 stop-server.bat（或关掉那个服务窗口），再重试。"
        ) from exc
    os.close(handle)


def _extract(archive: Path) -> None:
    """解压到 runtime/python（压缩包内顶层目录统一剥掉）。

    先解到 _staging，就绪后再把旧的换下来：换不上（被占用）就报错，
    绝不留下一个残缺的运行时。
    """
    staging = RUNTIME_DIR.parent / "_staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    print("  解压 ...")
    with tarfile.open(archive, "r:gz") as tar:
        members = []
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] in ("", ".", ".."):
                continue
            if member.name.startswith("/") or ".." in parts:
                continue
            member.name = str(Path(*parts[1:])) if len(parts) > 1 else ""
            if member.name:
                members.append(member)
        try:
            tar.extractall(staging, members=members, filter="data")
        except TypeError:  # Python < 3.12 没有 filter 参数
            tar.extractall(staging, members=members)

    RUNTIME_DIR.parent.mkdir(parents=True, exist_ok=True)
    retired = None
    if RUNTIME_DIR.exists():
        try:
            _assert_runtime_not_in_use()
        except RuntimeSetupError:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        retired = RUNTIME_DIR.with_name("_retired")
        shutil.rmtree(retired, ignore_errors=True)
        RUNTIME_DIR.rename(retired)

    shutil.move(str(staging), str(RUNTIME_DIR))
    if retired is not None:
        shutil.rmtree(retired, ignore_errors=True)


def ensure(verbose: bool = True) -> Path:
    """确保运行时可用；缺就下载解压。失败抛 RuntimeSetupError。"""
    if is_installed():
        if verbose:
            print(f"  运行时已就绪：{PY_EXE}")
        return PY_EXE

    entry = asset()
    if verbose:
        print(f"  项目自带 Python {entry.get('python', '')}，装到 {RUNTIME_DIR}")
        print("  （只管这个项目，不动你电脑里已有的 Python）")

    cache = RUNTIME_DIR.parent / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / entry["asset"]

    if archive.exists() and _sha256(archive) != entry["sha256"]:
        archive.unlink(missing_ok=True)

    if not archive.exists():
        _fetch(entry, cache)

    _extract(archive)
    archive.unlink(missing_ok=True)

    if not PY_EXE.exists():
        raise RuntimeSetupError(
            f"解压后没找到 {PY_EXE}，压缩包可能不完整。删掉 runtime/ 目录重试。"
        )

    MARKER_FILE.write_text(
        json.dumps(
            {"python": entry.get("python", ""), "sha256": entry["sha256"],
             "asset": entry["asset"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    if verbose:
        print(f"  运行时就绪：{PY_EXE}")
    return PY_EXE


def _retire_runtime() -> None:
    """删掉旧运行时。删不掉就明确报错，不留下半个运行时。"""
    if not RUNTIME_DIR.exists():
        return
    _assert_runtime_not_in_use()
    retired = RUNTIME_DIR.with_name("_retired")
    shutil.rmtree(retired, ignore_errors=True)
    if retired.exists():  # 上一次遗留的目录还占着，同样说明有东西在用
        raise RuntimeSetupError(
            f"{retired} 删不掉，先把所有服务窗口关掉，再删掉这个目录重试。"
        )
    RUNTIME_DIR.rename(retired)
    shutil.rmtree(retired, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="准备项目自带的 Python 运行时")
    parser.add_argument("--check", action="store_true", help="只看状态，不下载")
    parser.add_argument("--force", action="store_true", help="删除重装")
    parser.add_argument("--sources", action="store_true", help="只显示下载地址")
    args = parser.parse_args(argv)

    try:
        entry = asset()
    except RuntimeSetupError as exc:
        print(f"\n[装不上] {exc}\n", file=sys.stderr)
        return 1

    if args.sources:
        print(f"  平台：{platform_key()}")
        for url in source_urls(entry):
            print(f"  {url}")
        return 0

    if args.check:
        state = "已安装" if is_installed() else "未安装"
        print(f"  平台：{platform_key()}")
        print(f"  版本：Python {entry.get('python', '')}（{entry.get('release', '')}）")
        print(f"  状态：{state}（{PY_EXE}）")
        return 0 if is_installed() else 1

    if args.force:
        try:
            _retire_runtime()
        except RuntimeSetupError as exc:
            print(f"\n[装不上] {exc}\n", file=sys.stderr)
            return 1

    try:
        ensure()
    except RuntimeSetupError as exc:
        print(f"\n[装不上] {exc}\n", file=sys.stderr)
        return 1

    if args.force:
        print()
        print("  这是一份全新的运行时，里面的依赖是空的。")
        print("  接下来双击 start-server.bat（或跑 bootstrap.py）会把依赖装回来。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
