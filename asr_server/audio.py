"""音频解码：把任意常见格式读成单声道 float32。"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf


class AudioDecodeError(Exception):
    pass


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """字节流 -> (单声道 float32 波形, 采样率)。

    支持 wav / flac / ogg / mp3 等 libsndfile 能解的格式。
    """
    if not data:
        raise AudioDecodeError("音频内容为空")
    try:
        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # libsndfile 的报错类型不统一
        raise AudioDecodeError(
            f"无法解码音频（需要 wav/flac/ogg/mp3 等常见格式）：{exc}"
        ) from exc

    if samples.size == 0:
        raise AudioDecodeError("音频没有有效采样点")

    # 多声道取均值转单声道
    if samples.shape[1] > 1:
        samples = samples.mean(axis=1)
    else:
        samples = samples[:, 0]

    return np.ascontiguousarray(samples, dtype=np.float32), int(sample_rate)


def duration_seconds(samples: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(samples.shape[0]) / float(sample_rate)
