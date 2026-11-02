"""读取 config.toml，提供带默认值的配置对象。

配置来源优先级（后者覆盖前者）：
    代码默认值 < config.toml < 环境变量

环境变量用于 Docker / systemd / 临时调试，不必改文件：
    ASR_HOST  ASR_PORT  ASR_TOKEN  ASR_MODEL_DIR  ASR_DATA_DIR  ASR_CONFIG
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = ""
    model_type: str = "qwen3-asr-0.6b"
    model_dir: str = ""
    vad_model: str = ""
    vad_max_speech_seconds: int = 20
    num_threads: int = 6
    max_new_tokens: int = 512
    max_total_len: int = 2048
    hotwords: str = ""
    idle_unload_seconds: int = 300
    preload: bool = False


class ConfigError(Exception):
    """配置有问题，且能给出人能看懂的说明。"""


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def load_config(path: Path | None = None) -> Config:
    """读配置；文件不存在时返回默认值。"""
    from asr_server.paths import config_path

    if path is None:
        path = config_path()

    cfg = Config()
    if path.exists():
        try:
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(
                f"配置文件格式有问题，第 {getattr(exc, 'lineno', '?')} 行附近：{exc}\n"
                f"文件：{path}"
            ) from exc

        server = raw.get("server", {})
        model = raw.get("model", {})
        behavior = raw.get("behavior", {})
        cfg = Config(
            host=str(server.get("host", cfg.host)),
            port=int(server.get("port", cfg.port)),
            token=str(server.get("token", cfg.token) or ""),
            model_type=str(model.get("model_type", cfg.model_type)),
            model_dir=str(model.get("model_dir", cfg.model_dir) or ""),
            vad_model=str(model.get("vad_model", cfg.vad_model) or ""),
            vad_max_speech_seconds=int(
                model.get("vad_max_speech_seconds", cfg.vad_max_speech_seconds)
            ),
            num_threads=int(model.get("num_threads", cfg.num_threads)),
            max_new_tokens=int(model.get("max_new_tokens", cfg.max_new_tokens)),
            max_total_len=int(model.get("max_total_len", cfg.max_total_len)),
            hotwords=str(model.get("hotwords", cfg.hotwords) or ""),
            idle_unload_seconds=int(
                behavior.get("idle_unload_seconds", cfg.idle_unload_seconds)
            ),
            preload=bool(behavior.get("preload", cfg.preload)),
        )

    # 环境变量覆盖
    if (v := _env("ASR_HOST")) is not None:
        cfg.host = v
    if (v := _env("ASR_PORT")) is not None:
        try:
            cfg.port = int(v)
        except ValueError as exc:
            raise ConfigError(f"环境变量 ASR_PORT 不是数字：{v}") from exc
    if (v := _env("ASR_TOKEN")) is not None:
        cfg.token = v
    if (v := _env("ASR_MODEL_DIR")) is not None:
        cfg.model_dir = v

    # 绑非环回地址却不设令牌，等于把麦克风接口开放给整个局域网
    if not _is_loopback(cfg.host) and not cfg.token:
        raise ConfigError(
            f"host = \"{cfg.host}\" 表示允许别的机器访问，这时必须设置 token。\n"
            "在 config.toml 的 [server] 里填 token，或设环境变量 ASR_TOKEN。\n"
            "只想本机用就把 host 改回 \"127.0.0.1\"。"
        )

    return cfg
