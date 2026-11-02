"""一键启动：第一次自动装运行时、装依赖、下模型，之后直接起服务。

设计准则
--------
**只用项目自己的 Python**。不读 PATH、不读注册表、不找 conda、不建 .venv。
运行时在 ``runtime/python``（由 asr_server.runtime 下载，来源是 python-build-standalone，
和 uv 用的是同一份）。这样用户电脑里有没有 Python、装没装 conda，都影响不到本项目；
反过来本项目也不会往用户的 Python 里装任何东西。

下载失败就是失败，明确告诉用户怎么办，不偷偷改用系统 Python —— 一用就等于
把「能不能跑」交回给了外部环境。

用法：
    python bootstrap.py             安装（若需要）并启动服务
    python bootstrap.py --setup     只装依赖和模型，不起服务
    python bootstrap.py --check     检查环境状态
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from asr_server import runtime  # noqa: E402

IS_WIN = sys.platform == "win32"
RUNTIME_PY = runtime.runtime_python()

REQUIRED_MODULES = ["sherpa_onnx", "numpy", "fastapi", "uvicorn", "multipart", "soundfile"]
OPTIONAL_MODULES = ["sounddevice"]

PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_HOST = "pypi.tuna.tsinghua.edu.cn"


def say(msg: str = "") -> None:
    print(msg, flush=True)


def _run(py: str, code: str, timeout: int = 60) -> tuple[int, str]:
    """用指定解释器跑一段代码，返回 (退出码, 输出)。"""
    try:
        proc = subprocess.run(
            [py, "-c", code],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return proc.returncode, (proc.stdout or "").strip()
    except Exception as exc:
        return 1, f"{exc}"


def has_deps(py: str) -> bool:
    code = (
        "import importlib.util as u;"
        f"print(all(u.find_spec(m) for m in {REQUIRED_MODULES!r}))"
    )
    rc, out = _run(py, code)
    return rc == 0 and out.endswith("True")


def missing_optional(py: str) -> bool:
    code = (
        "import importlib.util as u;"
        f"print(all(u.find_spec(m) for m in {OPTIONAL_MODULES!r}))"
    )
    rc, out = _run(py, code)
    return not (rc == 0 and out.endswith("True"))


def runtime_version(py: str) -> str:
    rc, out = _run(py, "import sys;print('%d.%d.%d' % sys.version_info[:3])")
    return out if rc == 0 else "?"


def port_free(host: str, port: int) -> bool:
    import socket

    probe = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((probe, port)) != 0


def pip_install(py: str) -> bool:
    req_file = ROOT / "requirements.txt"
    base = [py, "-m", "pip", "install", "--disable-pip-version-check", "-q",
            "--no-input", "--no-warn-script-location", "--timeout", "60"]

    target = (["-r", str(req_file)] if req_file.exists() else [
        "sherpa-onnx==2.0.1", "numpy>=1.26", "fastapi>=0.110",
        "uvicorn>=0.27", "python-multipart>=0.0.9", "soundfile>=0.12",
    ])

    for label, mirror in (("清华镜像", True), ("官方源", False)):
        say(f"  用{label}安装 ...")
        extra = ["-i", PIP_INDEX, "--trusted-host", PIP_HOST] if mirror else []
        if subprocess.run(base + extra + target).returncode == 0:
            break
        say(f"  {label}失败，换一个")
    else:
        return False

    # 录音测试用的可选组件，装不上不影响服务
    if missing_optional(py):
        subprocess.run(base + ["sounddevice>=0.4.6"])
    return True


def banner(host: str, port: int) -> None:
    base = f"http://127.0.0.1:{port}" if host in ("0.0.0.0", "::") else f"http://{host}:{port}"
    line = "=" * 46
    say()
    say(line)
    say("  语音识别服务已启动")
    say(line)
    say(f"  地址    {base}")
    say("  模型    qwen3-asr-0.6b")
    say("  内存    用时约 1.1 GB，空闲 5 分钟后自动释放")
    say()
    say("  接口    POST /v1/audio/transcriptions")
    say(f"  文档    {base}/docs")
    say()
    say("  说话测试  另开一个窗口运行 test-server.bat")
    say("  停止服务  Ctrl+C，或直接关掉这个窗口")
    say(line)
    say()


def prepare_model(py: str) -> bool:
    """在子进程里跑下载，确保用的是装了依赖的那个解释器。"""
    code = (
        "import sys;"
        f"sys.path.insert(0,{str(ROOT)!r});"
        "from asr_server.download import ensure_model, ensure_vad;"
        "from asr_server.paths import model_dir, vad_model_path;"
        "ok = ensure_model(model_dir(), quiet=True);"
        "ensure_vad(vad_model_path()) if ok else None;"
        "sys.exit(0 if ok else 1)"
    )
    return subprocess.run([py, "-c", code], cwd=str(ROOT)).returncode == 0


def relaunch_under_runtime(argv: list[str]) -> int:
    """当前进程不是项目运行时（比如用户手动 python bootstrap.py）→ 换过去重跑。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["SELP_ASR_RELAUNCHED"] = "1"
    return subprocess.run([str(RUNTIME_PY), str(Path(__file__).resolve()), *argv],
                          cwd=str(ROOT), env=env).returncode


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="selp-asr 一键启动")
    parser.add_argument("--setup", action="store_true", help="只装运行时/依赖/模型，不启动")
    parser.add_argument("--check", action="store_true", help="只看环境状态")
    parser.add_argument("--reinstall", action="store_true", help="强制重装依赖")
    args = parser.parse_args(argv)

    # 被系统 Python 拉起来时，重跑的子进程会重新打印标题，外层就不重复了
    if not os.environ.get("SELP_ASR_RELAUNCHED"):
        say()
        say("selp-asr 本地语音识别")
        say("=" * 46)

    # --- 0. 换到项目自己的解释器 ---
    # 用户可能直接 `python bootstrap.py`，那个 python 是系统里的。先把运行时备好，
    # 然后立刻换过去重跑：绝不在系统 Python 里装包，也不让两边版本对不上。
    try:
        runtime.ensure(verbose=False)
    except runtime.RuntimeSetupError as exc:
        say()
        say(str(exc))
        return 1

    if Path(sys.executable).resolve() != RUNTIME_PY.resolve():
        # 重跑会重新打印标题，所以这里不再打印额外内容
        return relaunch_under_runtime(argv)

    # --- 1. 环境自检 ---
    say("[1/3] 检查运行环境")
    py = str(RUNTIME_PY)
    say(f"  Python：{py}（{runtime_version(py)}）")

    if args.check:
        from asr_server.download import TOTAL_BYTES, _human, verify_model_dir
        from asr_server.paths import model_dir

        problems = verify_model_dir(model_dir())
        try:
            from asr_server.config import load_config

            cfg = load_config()
            host, port = cfg.host, cfg.port
            say(f"  端口 {port} 可用：{port_free(host, port)}")
        except Exception as exc:
            say(f"  配置有问题：{exc}")
        say(f"  依赖完整：{has_deps(py)}")
        say(f"  录音组件：{not missing_optional(py)}")
        say(f"  模型目录：{model_dir()}")
        say(f"  模型完整：{not problems}（{_human(TOTAL_BYTES)}）")
        for p in problems:
            say(f"    - {p}")
        return 0

    if args.reinstall or not has_deps(py):
        say("  安装依赖（约 130 MB，只装进项目自己的 runtime，第一次会慢一点）")
        if not pip_install(py):
            say()
            say("  依赖安装失败。常见原因是网络不通。")
            say("  可以手动执行看真实报错：")
            say(f"    {py} -m pip install -r requirements.txt")
            return 1
        say("  依赖就绪")
    else:
        say("  依赖已就绪")

    # --- 2. 模型 ---
    say("[2/3] 检查模型文件")
    if not prepare_model(py):
        say()
        say("  模型没准备好，服务无法启动。")
        say("  可以重试一次，或按 README 第七节手动下载。")
        return 1

    if args.setup:
        say()
        say("环境准备完成。运行 start-server.bat 启动服务。")
        return 0

    # --- 3. 启动 ---
    say("[3/3] 启动服务")
    rc, out = _run(
        py,
        "import sys;"
        f"sys.path.insert(0,{str(ROOT)!r});"
        "from asr_server.config import load_config;"
        "c=load_config();"
        "print(f'{c.host}|{c.port}')",
    )
    if rc != 0:
        say()
        say(f"  配置读不出来：{out}")
        return 1
    host, port_s = out.split("|")
    port = int(port_s)

    if not port_free(host, port):
        say()
        say(f"  端口 {port} 已被占用，服务起不来。")
        say(f"  1) 可能已经在运行了：浏览器打开 http://127.0.0.1:{port}/health")
        say("  2) 想换端口：改 config.toml 的 port，或设环境变量 ASR_PORT=9000")
        if IS_WIN:
            say(f"  3) 想看是谁占着：命令行执行 netstat -ano | findstr {port}")
        else:
            say(f"  3) 想看是谁占着：lsof -i :{port}")
        return 1

    banner(host, port)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        return subprocess.run([py, "-m", "asr_server"], cwd=str(ROOT), env=env).returncode
    except KeyboardInterrupt:
        say("\n已停止。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
