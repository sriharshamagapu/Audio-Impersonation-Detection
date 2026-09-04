import numpy as np
import torch
import torchaudio


class ReplayDetector:
    """
    Lightweight replay/liveness detector.

    This detector looks for acoustic characteristics that can
    provide evidence of replayed/recorded audio.

    Output:
        replay_probability
        live_probability
        features
    """

    def __init__(self):
        print("Replay detector initialized.")

    def load_audio(self, wav_path, target_sr=16000):
        wav, sr = torchaudio.load(wav_path)

        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)

        # Resample
        if sr != target_sr:
            wav = torchaudio.functional.resample(
                wav,
                sr,
                target_sr
            )

        return wav.numpy(), target_sr

    def extract_features(self, audio, sample_rate):
        audio = np.asarray(audio, dtype=np.float32)

        # RMS energy
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))

        # Peak amplitude
        peak = float(np.max(np.abs(audio)) + 1e-12)

        # Clipping ratio
        clipping_ratio = float(
            np.mean(np.abs(audio) >= 0.99)
        )

        # FFT spectrum
        spectrum = np.abs(
            np.fft.rfft(audio)
        )

        frequencies = np.fft.rfftfreq(
            len(audio),
            1.0 / sample_rate
        )

        spectral_energy = spectrum ** 2

        total_energy = float(
            np.sum(spectral_energy) + 1e-12
        )

        # High-frequency energy above 6 kHz
        high_freq_mask = frequencies >= 6000

        high_freq_energy = float(
            np.sum(
                spectral_energy[high_freq_mask]
            )
            / total_energy
        )

        # Spectral centroid
        spectral_centroid = float(
            np.sum(
                frequencies * spectral_energy
            )
            / total_energy
        )

        return {
            "rms": rms,
            "peak": peak,
            "clipping_ratio": clipping_ratio,
            "high_frequency_energy": high_freq_energy,
            "spectral_centroid": spectral_centroid
        }

    def predict(self, wav_path):

        audio, sample_rate = self.load_audio(
            wav_path
        )

        features = self.extract_features(
            audio,
            sample_rate
        )

        # Start with a neutral replay score.
        replay_score = 0.0

        # Strong clipping can indicate recording/replay artifacts.
        if features["clipping_ratio"] > 0.01:
            replay_score += 0.25

        # Very low high-frequency content can indicate
        # bandwidth limitations introduced by a replay/recording chain.
        if features["high_frequency_energy"] < 0.01:
            replay_score += 0.20

        # Very low spectral centroid can also indicate
        # restricted/band-limited audio.
        if features["spectral_centroid"] < 1800:
            replay_score += 0.15

        # Cap heuristic score.
        replay_probability = min(
            replay_score,
            1.0
        )

        live_probability = 1.0 - replay_probability

        return {
            "replay_probability": replay_probability,
            "live_probability": live_probability,
            "features": features
        }


if __name__ == "__main__":

    print("=" * 50)
    print("REPLAY / LIVENESS DETECTOR TEST")
    print("=" * 50)

    detector = ReplayDetector()

    result = detector.predict(
        r"sample_audio\test.wav"
    )

    print()
    print("Replay probability :",
          result["replay_probability"])

    print("Live probability   :",
          result["live_probability"])

    print()
    print("Extracted features:")

    for key, value in result["features"].items():
        print(f"  {key}: {value}")

    print()
    print("REPLAY DETECTOR: OK")