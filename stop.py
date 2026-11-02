"""优雅停止本地语音服务：调 /v1/audio/shutdown，等进程真的退出。

退出码：
  0 已停止（含「本来就没在跑」）
  1 请求没被接受
  2 发了停止请求但端口仍被占用（服务没退干净）
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

ROOT = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from asr_server.config import ConfigError, load_config  # noqa: E402


def _port_open(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"配置有问题：{exc}")
        return 1

    probe = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
    url = f"http://{probe}:{cfg.port}/v1/audio/shutdown"

    if not _port_open(probe, cfg.port):
        print("服务未在运行")
        return 0

    request = urllib.request.Request(url, data=b"", method="POST")
    if cfg.token:
        request.add_header("Authorization", f"Bearer {cfg.token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        print(f"服务拒绝了停止请求（{exc.code}）：{detail}")
        return 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"无法连接服务：{exc}")
        return 1

    if not payload.get("shutting_down"):
        print(f"停止请求未被接受：{payload}")
        return 1

    # 等服务退出：端口关闭即认为停好，最多等 15 秒
    for _ in range(150):
        if not _port_open(probe, cfg.port):
            print("服务已停止")
            return 0
        time.sleep(0.1)

    print(f"已发送停止请求，但 {cfg.port} 端口仍被占用")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
