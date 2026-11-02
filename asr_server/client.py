"""HTTP 客户端：cli 与测试脚本共用。

走 HTTP 而不是进程内加载模型的原因：
服务已经占着 1.1 GB 内存，再加载一份就是 2.2 GB，CPU 机器上很难受。
而且这样测的是「服务能不能用」，正是要验证的事。
"""
from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscribeResult:
    text: str
    language: str
    duration: float
    elapsed: float


class ClientError(Exception):
    """连不上或服务返回错误。"""


def _base(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _headers(token: str) -> dict[str, str]:
    headers = {"User-Agent": "selp-asr/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def request(
    base_url: str,
    path: str,
    token: str = "",
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 30.0,
) -> bytes:
    headers = _headers(token)
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(
        _base(base_url, path), data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        if exc.code == 401:
            raise ClientError(
                "服务要求令牌（401）。检查 config.toml 的 token，留空表示不需要。"
            ) from exc
        raise ClientError(f"服务返回 {exc.code}：{detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise ClientError(
            f"连不上 {base_url}。先启动服务（start-server.bat），或检查地址对不对。\n"
            f"  底层错误：{exc.reason}"
        ) from exc


def health(base_url: str, token: str = "", timeout: float = 5.0) -> dict:
    return json.loads(request(base_url, "/health", token, timeout=timeout))


def warmup(base_url: str, token: str = "", timeout: float = 60.0) -> None:
    try:
        request(base_url, "/v1/audio/warmup", token, method="POST", timeout=timeout)
    except ClientError:
        pass  # 预热失败不影响正常转录，请求时自己会加载


def unload(base_url: str, token: str = "", timeout: float = 30.0) -> None:
    request(base_url, "/v1/audio/unload", token, method="POST", timeout=timeout)


def shutdown(base_url: str, token: str = "", timeout: float = 10.0) -> None:
    request(base_url, "/v1/audio/shutdown", token, method="POST", timeout=timeout)


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes]]) -> tuple[bytes, str]:
    boundary = "----selp-asr-" + uuid_hex()
    lines: list[bytes] = []
    for name, value in fields.items():
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        lines.append(value.encode("utf-8"))
        lines.append(b"\r\n")
    for name, (filename, data) in files.items():
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        lines.append(f"Content-Type: {ctype}\r\n\r\n".encode())
        lines.append(data)
        lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode())
    return b"".join(lines), f"multipart/form-data; boundary={boundary}"


def uuid_hex() -> str:
    import uuid

    return uuid.uuid4().hex


def transcribe_bytes(
    base_url: str,
    data: bytes,
    filename: str = "audio.wav",
    token: str = "",
    model: str = "qwen3-asr-0.6b",
    timeout: float = 120.0,
) -> TranscribeResult:
    body, ctype = _multipart(
        {"model": model}, {"file": (filename, data)}
    )
    raw = request(
        base_url,
        "/v1/audio/transcriptions",
        token,
        method="POST",
        body=body,
        content_type=ctype,
        timeout=timeout,
    )
    payload = json.loads(raw)
    return TranscribeResult(
        text=payload.get("text", ""),
        language=payload.get("language", ""),
        duration=float(payload.get("duration", 0.0)),
        elapsed=float(payload.get("elapsed", 0.0)),
    )


def transcribe_file(
    base_url: str, path: Path, token: str = "", model: str = "qwen3-asr-0.6b", timeout: float = 120.0
) -> TranscribeResult:
    return transcribe_bytes(
        base_url, path.read_bytes(), path.name, token, model, timeout
    )
