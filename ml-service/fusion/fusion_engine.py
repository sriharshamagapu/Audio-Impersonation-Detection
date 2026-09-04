class FusionEngine:
    """
    Combines the outputs of:
    1. AASIST
    2. SSL anti-deepfake detector
    3. Replay/Liveness detector
    """

    def __init__(self):
        # Weights for the three detectors
        self.aasist_weight = 0.50
        self.ssl_weight = 0.30
        self.replay_weight = 0.20

        print("Fusion engine initialized.")

    def fuse(self, aasist_result, ssl_result, replay_result):
        """
        Calculate a combined spoof/deepfake score.

        All inputs should contain probabilities/scores
        between 0 and 1.
        """

        aasist_spoof = float(aasist_result["spoof_probability"])
        ssl_fake = float(ssl_result["fake_probability"])
        replay_score = float(replay_result["replay_probability"])

        combined_score = (
            self.aasist_weight * aasist_spoof
            + self.ssl_weight * ssl_fake
            + self.replay_weight * replay_score
        )

        combined_score = max(0.0, min(1.0, combined_score))

        return {
            "aasist_spoof_probability": aasist_spoof,
            "ssl_fake_probability": ssl_fake,
            "replay_score": replay_score,
            "combined_spoof_score": combined_score
        }


if __name__ == "__main__":
    print("=" * 50)
    print("FUSION ENGINE TEST")
    print("=" * 50)

    # Test values from our three detectors
    aasist_result = {
        "spoof_probability": 0.0000125
    }

    ssl_result = {
        "fake_probability": 0.0195243
    }

    replay_result = {
        "replay_probability": 0.35
    }

    fusion = FusionEngine()

    result = fusion.fuse(
        aasist_result,
        ssl_result,
        replay_result
    )

    print()
    print("AASIST spoof score :", result["aasist_spoof_probability"])
    print("SSL fake score     :", result["ssl_fake_probability"])
    print("Replay score       :", result["replay_score"])
    print("Combined score     :", result["combined_spoof_score"])

    print()
    print("FUSION ENGINE: OK")