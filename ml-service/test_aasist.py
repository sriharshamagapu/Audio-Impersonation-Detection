import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing.audio import load_audio
from aasist.aasist_detector import AASISTDetector


audio_path = Path(__file__).resolve().parent.parent / "sample_audio" / "test.wav"

audio, sample_rate = load_audio(str(audio_path))

print("Audio loaded:", len(audio), "samples")
print("Sample rate:", sample_rate, "Hz")

detector = AASISTDetector()

result = detector.predict(torch.tensor(audio, dtype=torch.float32))

print("AASIST result:")
print("Bonafide probability:", result["bonafide_probability"])
print("Spoof probability:", result["spoof_probability"])