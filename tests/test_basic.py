"""回归测试：拿官方测试音频跑真实识别，断言不能退化。

这些测试需要模型文件（约 941 MB），没有模型时自动跳过，
所以 CI 里只跑不依赖模型的部分。

    python -m pytest tests -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asr_server import client  # noqa: E402
from asr_server.download import REQUIRED_FILES, TOTAL_BYTES, verify_model_dir  # noqa: E402
from asr_server.paths import model_dir, vad_model_path  # noqa: E402

MODEL_DIR = model_dir()
HAS_MODEL = not verify_model_dir(MODEL_DIR)
needs_model = pytest.mark.skipif(not HAS_MODEL, reason="模型未下载，跳过识别测试")


# ---------------- 不依赖模型的测试 ----------------

def test_version_and_imports():
    """包能导入，模块结构完整。"""
    import asr_server
    from asr_server import audio, config, download, engine, paths

    assert asr_server.__doc__
    assert audio and config and download and engine and paths


def test_required_files_sizes_are_plausible():
    """文件表里每个条目都有正整数字节数，防止写错零或负数。"""
    for name, size in REQUIRED_FILES.items():
        assert isinstance(size, int) and size > 0, name
    assert TOTAL_BYTES > 900 * 1024 * 1024


def test_config_defaults():
    """没配置文件时给出可用默认值。"""
    from asr_server.config import Config

    cfg = Config()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8765
    assert cfg.token == "", "默认不该带令牌，否则公开的默认值等于没有鉴权"
    assert cfg.idle_unload_seconds > 0


def test_config_rejects_open_host_without_token():
    """默认配置能正常加载；绑非环回地址却不要令牌时必须拒绝启动。"""
    import os

    from asr_server.config import ConfigError, load_config

    # 默认（环回 + 无令牌）不该抛
    assert load_config(Path("__not_exists__.toml")).host == "127.0.0.1"

    old_host = os.environ.get("ASR_HOST")
    old_token = os.environ.get("ASR_TOKEN")
    os.environ["ASR_HOST"] = "0.0.0.0"
    os.environ.pop("ASR_TOKEN", None)
    try:
        with pytest.raises(ConfigError):
            load_config(Path("__not_exists__.toml"))
        # 给了令牌就应该放行
        os.environ["ASR_TOKEN"] = "s3cret"
        assert load_config(Path("__not_exists__.toml")).token == "s3cret"
    finally:
        if old_host is None:
            os.environ.pop("ASR_HOST", None)
        else:
            os.environ["ASR_HOST"] = old_host
        if old_token is None:
            os.environ.pop("ASR_TOKEN", None)
        else:
            os.environ["ASR_TOKEN"] = old_token


def test_port_env_override():
    """ASR_PORT 能覆盖端口，非法值报可读错误。"""
    import os

    from asr_server.config import ConfigError, load_config

    old = os.environ.get("ASR_PORT")
    try:
        os.environ["ASR_PORT"] = "9001"
        assert load_config(Path("__not_exists__.toml")).port == 9001
        os.environ["ASR_PORT"] = "abc"
        with pytest.raises(ConfigError):
            load_config(Path("__not_exists__.toml"))
    finally:
        if old is None:
            os.environ.pop("ASR_PORT", None)
        else:
            os.environ["ASR_PORT"] = old


def test_multipart_builder():
    """手写的 multipart 报文结构正确（不依赖 requests）。"""
    from asr_server.client import _multipart

    body, ctype = _multipart({"model": "qwen3-asr-0.6b"}, {"file": ("a.wav", b"RIFFDATA")})
    assert ctype.startswith("multipart/form-data; boundary=")
    boundary = ctype.split("boundary=")[1]
    assert f"--{boundary}".encode() in body
    assert b'name="file"; filename="a.wav"' in body
    assert b"RIFFDATA" in body
    assert body.endswith(f"--{boundary}--\r\n".encode())


def test_audio_rejects_garbage():
    """乱码字节要给出可读错误，而不是崩栈。"""
    from asr_server.audio import AudioDecodeError, decode_audio

    with pytest.raises(AudioDecodeError):
        decode_audio(b"not an audio file")
    with pytest.raises(AudioDecodeError):
        decode_audio(b"")


# ---------------- 自带运行时 ----------------

def test_runtime_lock_is_consistent():
    """版本锁文件里的每个平台条目都有完整信息，且资产名带同一个版本号。"""
    from asr_server import runtime

    data = runtime.lock()
    version = data["python"]
    release = data["release"]

    assert version.startswith("3.")
    assert release.isdigit()
    assert data["sources"]["github"] and data["sources"]["nju"]
    assert "{release}" in data["sources"]["github"]

    platforms = data["platforms"]
    assert set(platforms) == {
        "windows-amd64", "windows-arm64",
        "linux-amd64", "linux-arm64",
        "macos-amd64", "macos-arm64",
    }
    for key, entry in platforms.items():
        assert entry["size"] > 1_000_000, key
        assert len(entry["sha256"]) == 64, key
        assert entry["sha256"] == entry["sha256"].lower(), key
        # 资产名和版本号必须对得上，改版本时漏改一个就会出现下载 404
        assert f"cpython-{version}+" in entry["asset"], key
        assert release in entry["asset"], key
        assert entry["asset"].endswith("install_only_stripped.tar.gz"), key


def test_runtime_platform_key_supported():
    """当前平台能识别出来（不能是 None），并能在 lock 里查到条目。"""
    from asr_server import runtime

    key = runtime.platform_key()
    assert key in runtime.lock()["platforms"]
    assert runtime.asset()["asset"]


def test_runtime_urls_point_to_asset():
    """下载地址拼出来的文件名正确，且两个源都在。"""
    from asr_server import runtime

    entry = runtime.asset()
    urls = runtime.source_urls(entry)
    assert len(urls) == 2
    for url in urls:
        assert url.startswith("https://")
        assert url.endswith("/" + entry["asset"])


def test_runtime_marker_written_after_complete_install():
    """已装好时，标记文件必须存在、内容和 lock 对得上。

    start-server.bat 靠这个文件区分「装好了」和「解压到一半被中断」，
    少了它就会去运行一个残缺的 python.exe（stdlib 不全，直接崩）。
    """
    import json

    from asr_server import runtime

    if not runtime.PY_EXE.exists():
        pytest.skip("运行时未安装")

    assert runtime.MARKER_FILE.exists(), "缺标记文件，start-server.bat 会当成没装好"
    marker = json.loads(runtime.MARKER_FILE.read_text(encoding="utf-8"))
    assert marker["sha256"] == runtime.asset()["sha256"]
    # 标准库必须真在，这是「解压完整」最直接的证据
    assert (runtime.RUNTIME_DIR / "Lib" / "encodings" / "__init__.py").exists()


def test_runtime_python_is_private():
    """项目跑的 Python 必须是自带的那份，不能是系统 Python。

    这是这个项目「不依赖用户环境」的根本: 它决定了 pip 装到哪里。
    """
    from asr_server import runtime

    if not runtime.is_installed():
        pytest.skip("运行时未安装")

    import subprocess

    out = subprocess.run(
        [str(runtime.runtime_python()), "-c",
         "import sys;print(sys.prefix);print(sys.base_prefix)"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert out.returncode == 0, out.stderr
    prefix, base_prefix = out.stdout.strip().splitlines()[:2]
    # base_prefix == prefix 说明它不是从别的 Python 借解释器的 venv 壳子，
    # 卸载/升级机器上的 Python 不会把它带走。
    assert prefix == base_prefix, f"运行时依赖了外部 Python：{base_prefix}"
    assert str(runtime.RUNTIME_DIR).lower() in prefix.lower()



# ---------------- 依赖模型的测试 ----------------

@needs_model
def test_model_dir_complete():
    assert not verify_model_dir(MODEL_DIR)


@needs_model
def test_vad_optional_exists():
    """VAD 是可选组件；存在就要大小对。"""
    if vad_model_path().exists():
        assert vad_model_path().stat().st_size == 643_854


@needs_model
def test_transcribe_fast1():
    """极快语速中文：断言关键短语，防止识别退化。"""
    from asr_server.audio import decode_audio
    from asr_server.engine import AsrEngine

    wav = MODEL_DIR / "test_wavs" / "fast1.wav"
    if not wav.exists():
        pytest.skip("示例音频缺失")
    samples, sr = decode_audio(wav.read_bytes())
    engine = AsrEngine(
        model_dir=MODEL_DIR,
        num_threads=6,
        max_new_tokens=512,
        max_total_len=2048,
        hotwords="",
        idle_unload_seconds=0,
        vad_model=vad_model_path(),
        vad_max_speech_seconds=20,
    )
    try:
        engine.warmup()
        result = engine.transcribe(samples, sr)
        text = result.text
        assert result.duration > 10
        assert result.elapsed > 0
        # 这几处是稳的，改错参数（比如 max_total_len 改小）会明显崩
        for word in ("左手", "轮胎", "树杈"):
            assert word in text, f"识别结果里没出现「{word}」：{text[:120]}"
    finally:
        engine.unload("test")


@needs_model
def test_transcribe_cantonese():
    """粤语 + 中英混读。"""
    from asr_server.audio import decode_audio
    from asr_server.engine import AsrEngine

    wav = MODEL_DIR / "test_wavs" / "cantonese.wav"
    if not wav.exists():
        pytest.skip("示例音频缺失")
    samples, sr = decode_audio(wav.read_bytes())
    engine = AsrEngine(
        model_dir=MODEL_DIR,
        num_threads=6,
        max_new_tokens=512,
        max_total_len=2048,
        hotwords="",
        idle_unload_seconds=0,
        vad_model=vad_model_path(),
        vad_max_speech_seconds=20,
    )
    try:
        engine.warmup()
        text = engine.transcribe(samples, sr).text
        assert "princess" in text.lower() or "公主" in text, text[:120]
    finally:
        engine.unload("test")


# ---------------- 服务端集成测试 ----------------

@pytest.fixture(scope="module")
def live_server():
    """真起一个服务，测完关掉。端口用 8799 避开日常用的 8765。"""
    import os
    import subprocess
    import time

    port = 8799
    env = os.environ.copy()
    env["ASR_PORT"] = str(port)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "-m", "asr_server"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                client.health(base, timeout=2)
                break
            except client.ClientError:
                time.sleep(0.5)
        else:
            proc.kill()
            out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            pytest.fail(f"服务没起来：{out[-500:]}")
        yield base
    finally:
        try:
            client.shutdown(base)
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_health_endpoint(live_server):
    info = client.health(live_server)
    assert info["status"] == "ok"
    assert info["model_type"] == "qwen3-asr-0.6b"
    assert "loaded" in info
    assert "num_threads" in info


def test_models_endpoint(live_server):
    import json

    payload = json.loads(client.request(live_server, "/v1/models"))
    assert payload["object"] == "list"
    assert any("qwen3-asr" in m["id"] for m in payload["data"])


@needs_model
def test_transcription_over_http(live_server):
    wav = MODEL_DIR / "test_wavs" / "raokouling.wav"
    if not wav.exists():
        pytest.skip("示例音频缺失")
    result = client.transcribe_file(live_server, wav)
    assert result.duration > 0
    assert result.elapsed > 0
    assert "广西" in result.text or "出租车" in result.text, result.text[:120]


def test_unload_then_reload(live_server):
    """卸载后还能再用（内存释放不会把服务搞坏）。"""
    client.warmup(live_server)
    client.unload(live_server)
    assert client.health(live_server)["loaded"] is False
    client.warmup(live_server)
    info = client.health(live_server)
    assert info["loaded"] is True
