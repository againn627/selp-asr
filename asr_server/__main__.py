"""python -m asr_server → 启动本地语音识别服务。

这里是启动入口，不用 uvicorn 命令行的原因：
/v1/audio/shutdown 需要拿到 uvicorn.Server 实例才能优雅退出。
"""
from __future__ import annotations

import socket
import sys


def main() -> int:
    from asr_server.config import ConfigError, load_config

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"\n[配置有问题] {exc}\n", file=sys.stderr)
        return 1

    probe = "127.0.0.1" if cfg.host in ("0.0.0.0", "::") else cfg.host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        busy = sock.connect_ex((probe, cfg.port)) == 0
    if busy:
        print(
            f"\n[启动失败] 端口 {cfg.port} 已被占用。\n"
            f"  1) 可能服务已经在跑了：浏览器打开 http://127.0.0.1:{cfg.port}/health\n"
            f"  2) 想换端口：改 config.toml 的 port，或设环境变量 ASR_PORT=9000\n"
            f"  3) 想知道是谁占着：netstat -ano | findstr {cfg.port}\n",
            file=sys.stderr,
        )
        return 1

    import uvicorn

    from asr_server import server as server_module

    # 用 Server 实例而不是 uvicorn.run：/v1/audio/shutdown 要拿它置 should_exit
    uvicorn_config = uvicorn.Config(
        server_module.app, host=cfg.host, port=cfg.port, log_level="info"
    )
    app_server = uvicorn.Server(uvicorn_config)
    server_module.app.state.server = app_server
    app_server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
