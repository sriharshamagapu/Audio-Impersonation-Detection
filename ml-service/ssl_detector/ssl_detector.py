import torch
import torchaudio

from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from huggingface_hub import PyTorchModelHubMixin


device = torch.device("cpu")


class SSLModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        cfg = Wav2Vec2Config(
            encoder_layers=12,
            encoder_embed_dim=768,
            quantize_targets=True,
            latent_dim=256,
            final_dim=256,
        )

        self.model = Wav2Vec2Model(cfg)

    def extract_feat(self, input_data):

        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]

        with torch.no_grad():
            features = self.model(
                input_data.to(device),
                mask=False,
                features_only=True
            )["x"]

        return features


class DeepfakeDetector(
    torch.nn.Module,
    PyTorchModelHubMixin
):

    def __init__(self):

        super().__init__()

        self.ssl_orig_output_dim = 768
        self.num_classes = 2

        self.m_ssl = SSLModel()

        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(
            output_size=1
        )

        self.proj_fc = torch.nn.Linear(
            in_features=768,
            out_features=2
        )

    def forward(self, wav):

        emb = self.m_ssl.extract_feat(wav)

        emb = emb.transpose(1, 2)

        pooled_emb = self.adap_pool1d(emb)

        pooled_emb = pooled_emb.squeeze(-1)

        logits = self.proj_fc(pooled_emb)

        return logits


class SSLDetector:

    def __init__(self):

        print("Loading SSL anti-deepfake model...")

        self.model = DeepfakeDetector.from_pretrained(
            "nii-yamagishilab/wav2vec-small-anti-deepfake"
        )

        self.model.to(device)
        self.model.eval()

        print("SSL model loaded successfully.")

    def load_audio(
        self,
        wav_path,
        target_sr=16000
    ):

        wav, sr = torchaudio.load(wav_path)

        # Convert stereo → mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)

        # Resample if required
        if sr != target_sr:
            wav = torchaudio.functional.resample(
                wav,
                sr,
                target_sr
            )

        # Layer normalization
        with torch.no_grad():
            wav = torch.nn.functional.layer_norm(
                wav,
                wav.shape
            )

        # Batch dimension
        return wav.unsqueeze(0).to(device)

    def predict(self, wav_path):

        wav = self.load_audio(wav_path)

        with torch.no_grad():

            logits = self.model(wav)

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

        fake_probability = float(
            probabilities[0, 0].item()
        )

        real_probability = float(
            probabilities[0, 1].item()
        )

        return {
            "fake_probability": fake_probability,
            "real_probability": real_probability
        }


if __name__ == "__main__":

    print("=" * 50)
    print("SSL ANTI-DEEPFAKE DETECTOR TEST")
    print("=" * 50)

    detector = SSLDetector()

    result = detector.predict(
        r"sample_audio\test.wav"
    )

    print()
    print("Fake probability :", result["fake_probability"])
    print("Real probability :", result["real_probability"])
    print()
    print("SSL DETECTOR: OK")