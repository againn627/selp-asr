"""Qwen3-ASR 引擎封装：懒加载 + 空闲卸载 + 单并发串行解码。"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("asr.engine")


@dataclass
class TranscribeResult:
    text: str
    language: str
    duration: float
    elapsed: float
    load_elapsed: float


class AsrEngine:
    def __init__(
        self,
        model_dir: Path,
        num_threads: int = 6,
        max_new_tokens: int = 512,
        max_total_len: int = 2048,
        hotwords: str = "",
        idle_unload_seconds: int = 300,
        sample_rate: int = 16000,
        vad_model: Path | None = None,
        vad_max_speech_seconds: int = 20,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.max_new_tokens = max_new_tokens
        self.max_total_len = max_total_len
        self.hotwords = hotwords
        self.sample_rate = sample_rate
        self.idle_unload_seconds = idle_unload_seconds
        self.vad_model = Path(vad_model) if vad_model else None
        self.vad_max_speech_seconds = vad_max_speech_seconds
        # 单次解码上限：max_total_len 装得下 prompt + 音频 token + 输出 token 时的长度
        # 2048 下约能盖住 100 秒音频，超过才走 VAD 分段
        self.single_pass_limit = 100.0

        self._vad = None
        self._recognizer = None
        self._lock = threading.Lock()          # 解码串行，避免多线程抢 CPU
        self._state_lock = threading.Lock()    # 保护 _recognizer 生命周期
        self._loading = False                  # 是否有线程正在加载模型
        self._last_used = 0.0
        self._loaded_at = 0.0
        self._stop = threading.Event()
        self._reaper: threading.Thread | None = None

    # ---------- 模型生命周期 ----------

    def _model_files(self) -> dict[str, Path]:
        return {
            "conv_frontend": self.model_dir / "conv_frontend.onnx",
            "encoder": self.model_dir / "encoder.int8.onnx",
            "decoder": self.model_dir / "decoder.int8.onnx",
            "tokenizer": self.model_dir / "tokenizer",
        }

    def check_model(self) -> list[str]:
        missing = [str(p) for p in self._model_files().values() if not p.exists()]
        return missing

    def _build(self):
        import sherpa_onnx

        files = self._model_files()
        logger.info("正在加载模型 %s（线程数=%d）", self.model_dir.name, self.num_threads)
        started = time.perf_counter()
        recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(files["conv_frontend"]),
            encoder=str(files["encoder"]),
            decoder=str(files["decoder"]),
            tokenizer=str(files["tokenizer"]),
            num_threads=self.num_threads,
            decoding_method="greedy_search",
            max_new_tokens=self.max_new_tokens,
            max_total_len=self.max_total_len,
            hotwords=self.hotwords,
            provider="cpu",
            debug=False,
        )
        cost = time.perf_counter() - started
        logger.info("模型加载完成，耗时 %.2fs", cost)
        self._loaded_at = time.monotonic()
        return recognizer, cost

    def _ensure_loaded(self) -> float:
        """确保模型在内存中，返回本次加载耗时（已加载则返回 0）。

        锁只保护「状态切换」，耗时的 _build() 在锁外跑：
        否则后台预加载会拿 _state_lock 足足 3 秒，/health 也跟着卡 3 秒，
        客户端探测就直接超时，看起来像「服务没起来」。
        并发调用（预加载线程 + warmup 请求）由 _loading 标记排队，只真正加载一次。
        """
        with self._state_lock:
            if self._recognizer is not None:
                return 0.0
            owner = not self._loading
            if owner:
                self._loading = True

        if not owner:
            # 别人正在加载：等它落地，不重复加载也不白等
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                with self._state_lock:
                    if self._recognizer is not None:
                        return 0.0
                    if not self._loading:
                        break
                time.sleep(0.05)
            raise RuntimeError("模型加载未完成")

        try:
            missing = self.check_model()
            if missing:
                raise FileNotFoundError(
                    "模型文件缺失：" + "；".join(missing) + "。请先运行 scripts/setup-model.ps1"
                )
            recognizer, cost = self._build()  # 耗时操作，锁外执行
            with self._state_lock:
                self._recognizer = recognizer
            return cost
        finally:
            with self._state_lock:
                self._loading = False

    def warmup(self) -> float:
        """确保模型已加载并跑一次极短解码，把首次推理的初始化开销前置。

        返回本次加载耗时（已加载则为 0）。已经加载时直接返回，不再跑解码：
        解码锁是单路串行的，已就绪时再占 0.5s 会拖慢紧接着的真实转写。
        加载失败照常抛出，由调用方决定怎么报。
        """
        load_cost = self._ensure_loaded()
        if load_cost == 0.0:
            return 0.0
        self.transcribe(np.zeros(self.sample_rate // 5, dtype=np.float32), self.sample_rate)
        return load_cost

    def unload(self, reason: str = "manual") -> bool:
        with self._state_lock:
            if self._recognizer is None:
                return False
            self._recognizer = None
        logger.info("已卸载模型（%s），内存已释放", reason)
        return True

    def _reap_loop(self) -> None:
        interval = max(5, min(60, self.idle_unload_seconds // 4))
        while not self._stop.wait(interval):
            if self.idle_unload_seconds <= 0:
                continue
            with self._state_lock:
                loaded = self._recognizer is not None
            if not loaded:
                continue
            if self._lock.locked():
                continue
            if time.monotonic() - self._last_used >= self.idle_unload_seconds:
                self.unload(f"空闲超过 {self.idle_unload_seconds}s")

    def start_reaper(self) -> None:
        if self.idle_unload_seconds <= 0 or self._reaper is not None:
            return
        self._reaper = threading.Thread(target=self._reap_loop, name="asr-reaper", daemon=True)
        self._reaper.start()

    def stop_reaper(self) -> None:
        self._stop.set()

    def _build_vad(self):
        import sherpa_onnx

        if self._vad is not None:
            return self._vad
        if not self.vad_model or not self.vad_model.exists():
            return None
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(self.vad_model)
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.35
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.max_speech_duration = float(self.vad_max_speech_seconds)
        cfg.sample_rate = self.sample_rate
        cfg.num_threads = 1
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
        return self._vad

    # ---------- 识别 ----------

    def _split_by_vad(self, samples: np.ndarray) -> list[np.ndarray]:
        """长音频按人声切段；VAD 不可用时退化为等长切分。"""
        vad = self._build_vad()
        window = 512
        if vad is not None:
            vad.reset()
            segments: list[np.ndarray] = []
            for start in range(0, len(samples), window):
                chunk = samples[start : start + window]
                if len(chunk) < window:
                    chunk = np.pad(chunk, (0, window - len(chunk)))
                vad.accept_waveform(chunk.astype(np.float32))
                while not vad.empty():
                    seg = vad.front
                    segments.append(np.asarray(seg.samples, dtype=np.float32).copy())
                    vad.pop()
            vad.flush()
            while not vad.empty():
                seg = vad.front
                segments.append(np.asarray(seg.samples, dtype=np.float32).copy())
                vad.pop()
            if segments:
                return segments

        # 退化：按 20 秒硬切
        step = int(self.sample_rate * 20)
        return [samples[i : i + step] for i in range(0, len(samples), step)]

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscribeResult:
        load_cost = self._ensure_loaded()
        started = time.perf_counter()
        with self._lock:
            recognizer = self._recognizer
            if recognizer is None:  # 并发卸载的极端情况
                raise RuntimeError("模型已被卸载，请重试")
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(stream)
            result = stream.result
            self._last_used = time.monotonic()
        elapsed = time.perf_counter() - started
        duration = float(samples.shape[0]) / float(sample_rate) if sample_rate else 0.0
        text = (result.text or "").strip()

        # 长音频：VAD 切段后逐段识别再拼接，避免超长输入被截断
        if duration > self.single_pass_limit:
            parts: list[str] = []
            for seg in self._split_by_vad(samples):
                if seg.size < self.sample_rate // 5:
                    continue
                with self._lock:
                    if self._recognizer is None:
                        raise RuntimeError("模型已被卸载，请重试")
                    stream = self._recognizer.create_stream()
                    stream.accept_waveform(sample_rate, seg)
                    self._recognizer.decode_stream(stream)
                    seg_text = (stream.result.text or "").strip()
                    self._last_used = time.monotonic()
                if seg_text:
                    parts.append(seg_text)
            if parts:
                text = "".join(parts)
            elapsed = time.perf_counter() - started

        return TranscribeResult(
            text=text,
            language=getattr(result, "lang", "") or "",
            duration=duration,
            elapsed=elapsed,
            load_elapsed=load_cost,
        )

    # ---------- 状态 ----------

    def status(self) -> dict:
        with self._state_lock:
            loaded = self._recognizer is not None
            loaded_at = self._loaded_at
        return {
            "loaded": loaded,
            "loaded_seconds": round(time.monotonic() - loaded_at, 1) if loaded else None,
            "idle_seconds": round(time.monotonic() - self._last_used, 1) if self._last_used else None,
            "model_dir": str(self.model_dir),
            "num_threads": self.num_threads,
            "max_new_tokens": self.max_new_tokens,
            "idle_unload_seconds": self.idle_unload_seconds,
        }
