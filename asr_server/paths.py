"""资源定位：找出配置、模型、日志该放哪。

三种运行方式都要能跑：
    pip install 后用 asr-server 命令   → 程序在 site-packages，可读不可写
    git clone 后 python -m asr_server  → 程序在仓库里，可读可写
    双击 start-server.bat             → 同仓库，可读可写

所以「写数据」的目录永远不在程序目录里，除非是便携模式（仓库里有 config.toml）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "selp-asr"

# 模型目录的相对位置（相对数据根）
MODEL_SUBDIR = "models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
VAD_SUBDIR = "models/silero_vad.onnx"


def program_root() -> Path:
    """程序所在目录：能放 config.toml 的地方。

    - git clone / 双击脚本：仓库根目录
    - pip 安装：site-packages 上一层（不往里写东西）
    """
    return Path(__file__).resolve().parent.parent


def is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def data_root() -> Path:
    """数据根目录：模型、配置的默认落脚点。

    优先顺序：
    1. 环境变量 ASR_DATA_DIR
    2. 程序目录（仅当可写，即便携模式/git clone）
    3. %LOCALAPPDATA%\\selp-asr（Windows）/ ~/.local/share/selp-asr（Linux/macOS）
    """
    env = os.environ.get("ASR_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()

    root = program_root()
    # pip 安装时程序目录在 site-packages 里，不往那儿写
    if "site-packages" not in str(root) and is_writable(root):
        return root

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    return base / APP_NAME


def config_path() -> Path:
    """配置文件路径。

    ASR_CONFIG 指定优先；否则程序目录有 config.toml 就用它（便携模式），
    再否则用数据目录下的 config.toml（可能是默认值，允许不存在）。
    """
    env = os.environ.get("ASR_CONFIG")
    if env:
        return Path(env).expanduser().resolve()

    portable = program_root() / "config.toml"
    if portable.exists():
        return portable
    return data_root() / "config.toml"


def model_dir() -> Path:
    """模型目录。"""
    env = os.environ.get("ASR_MODEL_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return data_root() / MODEL_SUBDIR


def vad_model_path() -> Path:
    """VAD 模型文件路径（长音频分段用，缺失不影响短句）。"""
    env = os.environ.get("ASR_VAD_MODEL")
    if env:
        return Path(env).expanduser().resolve()
    return data_root() / VAD_SUBDIR
