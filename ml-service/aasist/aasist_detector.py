import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Official AASIST code
OFFICIAL_DIR = Path(__file__).resolve().parent / "official"
sys.path.insert(0, str(OFFICIAL_DIR))

from models.AASIST import Model


class AASISTDetector:
    def __init__(self):
        self.device = torch.device("cpu")

        self.config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
        }

        self.model = Model(self.config).to(self.device)

        checkpoint = (
            Path(__file__).resolve().parent
            / "official"
            / "models"
            / "weights"
            / "AASIST.pth"
        )

        state_dict = torch.load(
            checkpoint,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _prepare_audio(self, audio):
        audio = audio.flatten()
        if not isinstance(audio, torch.Tensor):
           audio = torch.tensor(audio, dtype=torch.float32)
        target_length = self.config["nb_samp"]

        if len(audio) < target_length:
            audio = F.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        return audio.unsqueeze(0).to(self.device)

    def predict(self, audio):
        audio = self._prepare_audio(audio)

        with torch.no_grad():
            _, output = self.model(audio)

        probabilities = torch.softmax(output, dim=1)[0]

        return {
            "spoof_probability": float(probabilities[1].item()),
            "bonafide_probability": float(probabilities[0].item()),
        }


if __name__ == "__main__":
    print("AASIST detector module: OK")