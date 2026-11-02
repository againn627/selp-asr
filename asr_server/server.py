"""本地语音转文字服务（OpenAI 兼容接口）。

接口：
  GET  /health                       健康检查 + 模型状态
  GET  /v1/models                    模型列表（OpenAI 同构占位）
  POST /v1/audio/warmup              加载模型并热身（客户端可在录音的同时调）
  POST /v1/audio/transcriptions      multipart: file=音频字节, 可选 model / response_format
  POST /v1/audio/unload              只卸载模型释放内存，进程继续服务
  POST /v1/audio/shutdown            优雅退出整个服务
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asr_server.audio import AudioDecodeError, decode_audio  # noqa: E402
from asr_server.config import load_config  # noqa: E402
from asr_server.engine import AsrEngine  # noqa: E402
from asr_server.paths import model_dir, vad_model_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("asr.server")

cfg = load_config()
engine = AsrEngine(
    model_dir=model_dir(),
    num_threads=cfg.num_threads,
    max_new_tokens=cfg.max_new_tokens,
    max_total_len=cfg.max_total_len,
    hotwords=cfg.hotwords,
    idle_unload_seconds=cfg.idle_unload_seconds,
    vad_model=vad_model_path(),
    vad_max_speech_seconds=cfg.vad_max_speech_seconds,
)


def _preload() -> None:
    """把模型加载并热身。

    两种用法：
    * preload = true：在启动阶段同步跑，服务能接收请求时模型已就绪，代价是启动多等 3-4s。
    * preload = false（默认）：启动后在后台线程跑，uvicorn 立刻能接收请求；
      客端按麦克风时正好与录音、说话重叠，短句也不卡。
    """
    try:
        if engine.check_model():
            return
        started = time.perf_counter()
        engine.warmup()
        logger.info("模型已预先加载并热身，耗时 %.2fs", time.perf_counter() - started)
    except Exception:  # 预热失败不影响服务启动，下次请求会再试
        logger.exception("模型预热失败")


def _preload_in_background() -> None:
    """后台加载，不阻塞启动；不会与 warmup 请求重复加载（engine 内部有状态锁）。"""
    threading.Thread(target=_preload, name="asr-preload", daemon=True).start()


@asynccontextmanager
async def lifespan(_: FastAPI):
    missing = engine.check_model()
    if missing:
        logger.error("模型文件缺失：%s", "；".join(missing))
        logger.error("请运行 bootstrap.py 下载模型：python bootstrap.py")
    else:
        logger.info("模型目录就绪：%s", engine.model_dir)
    engine.start_reaper()
    logger.info("服务地址 http://%s:%s", cfg.host, cfg.port)
    if cfg.preload:
        _preload()
    else:
        # 不等它：服务立即可以接收请求，模型在后台加载（约 3s）
        _preload_in_background()
    yield
    engine.stop_reaper()
    engine.unload("服务退出")


app = FastAPI(title="Local ASR Server", version="1.0.0", lifespan=lifespan)
# 桌面前端（file:// 与 Vite 开发地址）需要直连转写接口，否则会被浏览器 CORS 预检拦下。
# 只放开本机来源，不对外网开放。
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(file://.*|http://(localhost|127\.0\.0\.1)(:\d+)?)$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(authorization: str | None = Header(default=None)) -> None:
    if not cfg.token:
        return
    expected = f"Bearer {cfg.token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="无效的认证信息")


@app.get("/health")
def health() -> dict:
    status = engine.status()
    return {
        "status": "ok" if not engine.check_model() else "model_missing",
        "model_type": cfg.model_type,
        **status,
    }


@app.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": cfg.model_type,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/audio/warmup")
def warmup(_: None = Depends(verify_token)) -> dict:
    """加载模型并热身，返回时就能立即转写。

    客户端在开始录音时就发这个请求，加载耗时与用户说话重叠，松手即出字。
    已经加载时直接返回，开销只有一次极短解码。
    """
    missing = engine.check_model()
    if missing:
        raise HTTPException(status_code=503, detail="模型文件缺失：" + "；".join(missing))
    started = time.perf_counter()
    load_cost = engine.warmup()
    return {
        "ready": True,
        "already_loaded": load_cost == 0.0,
        "load_seconds": round(load_cost, 2),
        "elapsed": round(time.perf_counter() - started, 2),
    }


@app.post("/v1/audio/unload")
def unload(_: None = Depends(verify_token)) -> dict:
    return {"unloaded": engine.unload("手动请求")}


@app.post("/v1/audio/shutdown")
def shutdown(_: None = Depends(verify_token)) -> dict:
    """优雅退出：稍后让 uvicorn 停止接收请求，退出路径上再卸模型。"""
    server = getattr(app.state, "server", None)

    def _stop() -> None:
        time.sleep(0.2)
        if server is not None:
            server.should_exit = True
        else:
            os._exit(0)

    threading.Thread(target=_stop, name="asr-shutdown", daemon=True).start()
    logger.info("收到停止请求，即将退出")
    return {"shutting_down": True}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    response_format: str = Form(default="json"),
    language: str = Form(default=""),
    prompt: str = Form(default=""),
    _: None = Depends(verify_token),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="音频文件为空")

    try:
        samples, sample_rate = decode_audio(data)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = engine.transcribe(samples, sample_rate)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # 推理异常
        logger.exception("识别失败")
        raise HTTPException(status_code=500, detail=f"识别失败：{exc}") from exc

    logger.info(
        "识别完成：音频 %.2fs，耗时 %.2fs，RTF %.2f，文本 %d 字",
        result.duration,
        result.elapsed,
        result.elapsed / result.duration if result.duration else 0,
        len(result.text),
    )

    if response_format == "text":
        return PlainTextResponse(result.text)
    if response_format in ("verbose_json", "srt", "vtt"):
        return JSONResponse(
            {
                "task": "transcribe",
                "language": result.language,
                "duration": round(result.duration, 3),
                "text": result.text,
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": round(result.duration, 3),
                        "text": result.text,
                    }
                ],
            }
        )
    return {
        "text": result.text,
        "language": result.language,
        "duration": round(result.duration, 3),
        "elapsed": round(result.elapsed, 3),
    }


if __name__ == "__main__":
    # 启动逻辑在 asr_server.__main__ 里（端口检查 + 优雅退出），这里只转发
    from asr_server.__main__ import main

    raise SystemExit(main())
