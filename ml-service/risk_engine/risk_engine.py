class RiskEngine:
    """
    Converts the fused spoof score into a risk decision.

    Levels:
        LOW
        VERIFY
        HIGH
        CRITICAL
    """

    def __init__(self):
        print("Risk engine initialized.")

    def assess(self, fusion_result):
        combined_score = float(
            fusion_result["combined_spoof_score"]
        )

        # Risk thresholds
        if combined_score < 0.20:
            risk_level = "LOW"
            action = "Allow"
        elif combined_score < 0.50:
            risk_level = "VERIFY"
            action = "Request additional verification"
        elif combined_score < 0.80:
            risk_level = "HIGH"
            action = "Block or require strong verification"
        else:
            risk_level = "CRITICAL"
            action = "Block immediately"

        # Calculate a simple uncertainty indicator.
        # This represents how close the score is to a decision boundary.
        boundaries = [0.20, 0.50, 0.80]
        distances = [
            abs(combined_score - boundary)
            for boundary in boundaries
        ]

        nearest_boundary_distance = min(distances)

        uncertainty = max(
            0.0,
            min(1.0, 1.0 - (nearest_boundary_distance / 0.20))
        )

        return {
            "risk_level": risk_level,
            "action": action,
            "combined_spoof_score": combined_score,
            "uncertainty": uncertainty,
            "detector_scores": {
                "aasist": fusion_result[
                    "aasist_spoof_probability"
                ],
                "ssl": fusion_result[
                    "ssl_fake_probability"
                ],
                "replay": fusion_result[
                    "replay_score"
                ]
            }
        }


if __name__ == "__main__":
    print("=" * 50)
    print("RISK ENGINE TEST")
    print("=" * 50)

    # Current Fusion Engine result
    fusion_result = {
        "aasist_spoof_probability": 0.0000125,
        "ssl_fake_probability": 0.0195243,
        "replay_score": 0.35,
        "combined_spoof_score": 0.07586354
    }

    risk_engine = RiskEngine()

    result = risk_engine.assess(fusion_result)

    print()
    print("Combined spoof score :", result["combined_spoof_score"])
    print("Risk level           :", result["risk_level"])
    print("Action               :", result["action"])
    print("Uncertainty          :", result["uncertainty"])

    print()
    print("Detector scores:")
    for name, score in result["detector_scores"].items():
        print(f"  {name}: {score}")

    print()
    print("RISK ENGINE: OK")