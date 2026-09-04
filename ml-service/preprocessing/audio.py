from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(file_path: str):
    """
    Load an audio file and return:
        waveform: numpy array
        sample_rate: integer
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    waveform, sample_rate = sf.read(path, always_2d=False)

    # Convert stereo/multi-channel audio to mono
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    waveform = waveform.astype(np.float32)

    return waveform, sample_rate


def normalize_audio(waveform: np.ndarray):
    """
    Normalize audio amplitude to approximately [-1, 1].
    """
    if waveform.size == 0:
        raise ValueError("Audio waveform is empty.")

    peak = np.max(np.abs(waveform))

    if peak > 0:
        waveform = waveform / peak

    return waveform.astype(np.float32)
def resample_audio(
    waveform: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int = 16000,
):
    """
    Resample a mono waveform to the target sample rate.
    """

    if waveform.size == 0:
        raise ValueError("Audio waveform is empty.")

    if original_sample_rate <= 0:
        raise ValueError("Original sample rate must be positive.")

    if target_sample_rate <= 0:
        raise ValueError("Target sample rate must be positive.")

    if original_sample_rate == target_sample_rate:
        return waveform.astype(np.float32)

    duration = len(waveform) / original_sample_rate
    target_length = int(round(duration * target_sample_rate))

    if target_length <= 0:
        raise ValueError("Audio is too short to resample.")

    old_positions = np.linspace(
        0,
        duration,
        num=len(waveform),
        endpoint=False,
    )

    new_positions = np.linspace(
        0,
        duration,
        num=target_length,
        endpoint=False,
    )

    resampled = np.interp(
        new_positions,
        old_positions,
        waveform,
    )

    return resampled.astype(np.float32)