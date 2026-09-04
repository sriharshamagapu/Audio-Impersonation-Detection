import sys
import os

# Allow imports from ml-service
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.audio import load_audio, normalize_audio, resample_audio
from aasist.aasist_detector import AASISTDetector
from ssl_detector.ssl_detector import SSLDetector
from replay.replay_detector import ReplayDetector
from fusion.fusion_engine import FusionEngine
from risk_engine.risk_engine import RiskEngine


class VoiceCloneGuardPipeline:

    def __init__(self):
        print("=" * 60)
        print("INITIALIZING VOICE CLONE GUARD")
        print("=" * 60)

        print("\n[1/5] Loading AASIST...")
        self.aasist = AASISTDetector()

        print("\n[2/5] Loading SSL detector...")
        self.ssl = SSLDetector()

        print("\n[3/5] Loading Replay detector...")
        self.replay = ReplayDetector()

        print("\n[4/5] Loading Fusion engine...")
        self.fusion = FusionEngine()

        print("\n[5/5] Loading Risk engine...")
        self.risk = RiskEngine()

        print("\n" + "=" * 60)
        print("VOICE CLONE GUARD INITIALIZED")
        print("=" * 60)

    def analyze(self, audio_path):

        print("\n" + "=" * 60)
        print("ANALYZING AUDIO")
        print("=" * 60)

        # --------------------------------------------------
        # AUDIO PREPROCESSING
        # --------------------------------------------------

        print("\n[PREPROCESSING]")

        audio, sample_rate = load_audio(audio_path)

        print(
            f"Loaded audio: {len(audio)} samples @ "
            f"{sample_rate} Hz"
        )

        audio = normalize_audio(audio)

        print("Audio normalization: OK")

        audio = resample_audio(
            audio,
            sample_rate,
            16000
        )

        print(
            f"Audio resampling: OK "
            f"({len(audio)} samples @ 16000 Hz)"
        )

        # --------------------------------------------------
        # AASIST
        # --------------------------------------------------

        print("\n[AASIST]")

        aasist_result = self.aasist.predict(audio)

        print(
            "Spoof probability:",
            aasist_result["spoof_probability"]
        )

        print(
            "Bonafide probability:",
            aasist_result["bonafide_probability"]
        )

        # --------------------------------------------------
        # SSL DETECTOR
        # --------------------------------------------------

        print("\n[SSL DETECTOR]")

        ssl_result = self.ssl.predict(audio_path)

        print(
            "Fake probability:",
            ssl_result["fake_probability"]
        )

        print(
            "Real probability:",
            ssl_result["real_probability"]
        )

        # --------------------------------------------------
        # REPLAY / LIVENESS
        # --------------------------------------------------

        print("\n[REPLAY / LIVENESS]")

        replay_result = self.replay.predict(audio_path)

        print(
            "Replay score:",
            replay_result["replay_probability"]
        )

        print(
            "Live score:",
            replay_result["live_probability"]
        )

        # --------------------------------------------------
        # FUSION
        # --------------------------------------------------

        print("\n[FUSION]")

        fusion_result = self.fusion.fuse(
            aasist_result,
            ssl_result,
            replay_result
        )

        print(
            "Combined spoof score:",
            fusion_result["combined_spoof_score"]
        )

        # --------------------------------------------------
        # RISK ENGINE
        # --------------------------------------------------

        print("\n[RISK ENGINE]")

        risk_result = self.risk.assess(
            fusion_result
        )

        print(
            "Risk level:",
            risk_result["risk_level"]
        )

        print(
            "Action:",
            risk_result["action"]
        )

        print(
            "Uncertainty:",
            risk_result["uncertainty"]
        )

        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        final_result = {
            "audio_file": audio_path,

            "aasist": aasist_result,

            "ssl": ssl_result,

            "replay": replay_result,

            "fusion": fusion_result,

            "risk": risk_result
        }

        return final_result


if __name__ == "__main__":

    audio_path = r"sample_audio\test.wav"

    pipeline = VoiceCloneGuardPipeline()

    result = pipeline.analyze(audio_path)

    print("\n" + "=" * 60)
    print("FINAL VOICE CLONE GUARD RESULT")
    print("=" * 60)

    print("\nRisk Level:")
    print(result["risk"]["risk_level"])

    print("\nAction:")
    print(result["risk"]["action"])

    print("\nCombined Spoof Score:")
    print(result["risk"]["combined_spoof_score"])

    print("\nUncertainty:")
    print(result["risk"]["uncertainty"])

    print("\n" + "=" * 60)
    print("END-TO-END PIPELINE: OK")
    print("=" * 60)