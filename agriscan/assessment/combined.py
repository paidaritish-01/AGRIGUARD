"""
Combined crop-health assessment service.
Synthesizes visual leaf classifications and historical environmental sensor data
into an integrated, calibrated diagnostic assessment.

Enforces:
- Independent evaluation of image and sensor streams.
- Explicit concordance vs discordance analysis without arbitrary score multiplication.
- Transparent reporting of uncertainties, missing modules, and domain boundaries.
- No pesticide recommendations.
"""

import time
from typing import Optional, Dict, Any, List
from agriscan.sensors.schema import SensorAnalysisResponse


STANDARD_COMBINED_LIMITATIONS = [
    "Combined assessments integrate preliminary visual indications with microclimate favorability; neither stream constitutes a laboratory or molecular disease confirmation.",
    "Do not treat models trained on close-up leaf photographs as validated for aerial imagery.",
    "Microclimate sensors reflect ambient localized conditions and may differ from internal leaf boundary layer conditions.",
    "Automated findings are preliminary screening indicators only. Consult a certified agronomist and conduct ground-truthing before taking agronomic action.",
    "Strict safety policy: AgriScan does not generate pesticide recommendations.",
]


class CombinedAssessmentCoordinator:
    """
    Coordinates multi-modal assessment by integrating image inference and sensor history.
    """

    def synthesize(
        self,
        image_id: Optional[str] = None,
        crop: str = "mango",
        image_result: Optional[Dict[str, Any]] = None,
        sensor_result: Optional[SensorAnalysisResponse] = None,
    ) -> Dict[str, Any]:
        """
        Merges image analysis and sensor analysis into a unified response contract.
        """
        t0 = time.time()
        effective_image_id = image_id or (image_result.get("image_id") if image_result else "unspecified_image")

        # 1. Standardize Image Analysis Block
        if image_result is not None:
            img_status = image_result.get("status", "uncertain")
            img_label = image_result.get("predicted_label")
            img_score = image_result.get("model_score")
            img_version = image_result.get("model_version")
            img_score_type = image_result.get("score_type", "raw_softmax_confidence")
            img_boxes = image_result.get("bounding_boxes")
            img_gps = image_result.get("gps_metadata")
        else:
            img_status = "not_assessed"
            img_label = None
            img_score = None
            img_version = None
            img_score_type = None
            img_boxes = None
            img_gps = None

        image_analysis_block = {
            "status": img_status,
            "predicted_class": img_label,
            "scores": img_score,
            "score_type": img_score_type,
            "model_version": img_version,
            "bounding_boxes": img_boxes,
            "gps_metadata": img_gps,
        }

        # 2. Standardize Sensor Analysis Block
        if sensor_result is not None:
            sns_status = sensor_result.status
            sns_quality = sensor_result.data_quality.model_dump()
            sns_observations = sensor_result.observations
            sns_derived = sensor_result.derived_features.model_dump() if sensor_result.derived_features else None
            sns_risks = [r.model_dump() for r in sensor_result.risk_assessment]
            sns_stresses = sensor_result.general_stress_indicators
        else:
            sns_status = "not_assessed"
            sns_quality = {
                "record_count": 0,
                "valid_count": 0,
                "missing_fields": [],
                "warnings": ["No sensor records provided or referenced."],
                "is_synthetic_fixture": False,
            }
            sns_observations = []
            sns_derived = None
            sns_risks = None
            sns_stresses = []

        sensor_analysis_block = {
            "status": sns_status,
            "data_quality": sns_quality,
            "observations": sns_observations,
            "derived_features": sns_derived,
            "risk_assessment": sns_risks,
            "general_stress_indicators": sns_stresses,
        }

        # 3. Combined Assessment Synthesis
        combined_status, concordance, explanation, recommendations = self._evaluate_evidence(
            img_status=img_status,
            img_label=img_label,
            img_score=img_score,
            sns_status=sns_status,
            sns_risks=sns_risks,
            sns_stresses=sns_stresses,
            crop=crop,
        )

        elapsed_ms = round((time.time() - t0) * 1000.0, 2)

        return {
            "image_id": effective_image_id,
            "crop": crop,
            "assessment_status": combined_status,
            "image_analysis": image_analysis_block,
            "sensor_analysis": sensor_analysis_block,
            "combined_assessment": {
                "status": combined_status,
                "evidence_concordance": concordance,
                "explanation": explanation,
                "recommended_actions": recommendations,
            },
            "limitations": STANDARD_COMBINED_LIMITATIONS,
            "processing_time_ms": elapsed_ms,
        }

    def _evaluate_evidence(
        self,
        img_status: str,
        img_label: Optional[str],
        img_score: Optional[float],
        sns_status: str,
        sns_risks: Optional[List[Dict[str, Any]]],
        sns_stresses: List[str],
        crop: str,
    ) -> Tuple[str, str, str, List[str]]:
        """
        Determines overall assessment status, concordance, explanation, and next actions.
        """
        # Case A: Neither module available
        if img_status in {"not_assessed", "model_unavailable", "failed"} and sns_status in {"not_assessed", "insufficient_data"}:
            return (
                "insufficient_data",
                "insufficient",
                "A validated image model or adequate sensor history is not available.",
                [
                    "Upload high-resolution close-up leaf imagery with adequate lighting.",
                    "Provide at least 24-72 hours of valid plot sensor history (temperature and relative humidity).",
                ],
            )

        # Case B: Image only available
        if img_status == "success" and sns_status in {"not_assessed", "insufficient_data"}:
            label_display = img_label.replace("_", " ").title() if img_label else "Unknown"
            score_display = f"{img_score:.2f}" if img_score is not None else "N/A"
            return (
                "partial_evidence",
                "single_source",
                f"Assessment based solely on visual close-up leaf imagery: suspected {label_display} (confidence: {score_display}). Environmental sensor history was unavailable.",
                [
                    "Conduct ground-truthing inspection on the affected tree canopy.",
                    "Connect microclimatic sensor history to evaluate disease favorability trends.",
                ],
            )

        # Case C: Sensor only available
        if sns_status == "assessed" and img_status in {"not_assessed", "model_unavailable", "failed"}:
            risk_summaries = []
            if sns_risks:
                for r in sns_risks:
                    if r.get("risk_level") in {"high", "moderate"}:
                        risk_summaries.append(f"{r['disease_or_condition']} ({r['risk_level']} risk)")

            risk_text = ", ".join(risk_summaries) if risk_summaries else "Low pathogen microclimate favorability"
            stress_text = f" Environmental stressors: {'; '.join(sns_stresses)}." if sns_stresses else ""
            img_note = " Image model unavailable." if img_status == "model_unavailable" else " No image supplied."

            return (
                "partial_evidence",
                "single_source",
                f"Assessment based solely on environmental microclimate history.{img_note} Sensor conditions indicate: {risk_text}.{stress_text}",
                [
                    "Scout orchard for early visual foliar or blossom symptoms corresponding to elevated risks.",
                    "Upload close-up leaf photographs once a validated vision model checkpoint is installed.",
                ],
            )

        # Case D: Both image and sensor modules produced results
        if img_status == "success" and sns_status == "assessed" and sns_risks:
            clean_label = (img_label or "").lower()
            anth_risk = next((r for r in sns_risks if "anthracnose" in r["disease_or_condition"].lower()), None)
            pm_risk = next((r for r in sns_risks if "powdery mildew" in r["disease_or_condition"].lower()), None)

            anth_level = anth_risk.get("risk_level", "low") if anth_risk else "low"
            pm_level = pm_risk.get("risk_level", "low") if pm_risk else "low"

            # Check Anthracnose concordance
            if "anthracnose" in clean_label:
                if anth_level in {"high", "moderate"}:
                    return (
                        "completed",
                        "concordant",
                        f"Concordant positive finding: Visual leaf symptoms indicate suspected Anthracnose, and recent sensor history confirms {anth_level} microclimatic favorability for Colletotrichum germination.",
                        [
                            "Inspect affected trees and neighboring canopies for foliar and twig dieback.",
                            "Consult an agronomist to evaluate appropriate cultural practices or canopy pruning.",
                        ],
                    )
                else:
                    return (
                        "completed",
                        "discordant",
                        f"Discordant finding: Visual symptoms suggest Anthracnose, but recent sensor history shows low microclimate favorability. Lesions may be inactive historical damage from an earlier wet cycle, or represent non-pathogenic leaf scorch.",
                        [
                            "Inspect whether leaf lesions are active or dry/calloused.",
                            "Review weather history beyond the current sensor window to identify prior infection events.",
                        ],
                    )

            # Check Powdery Mildew concordance
            elif "powdery_mildew" in clean_label or "mildew" in clean_label:
                if pm_level in {"high", "moderate"}:
                    return (
                        "completed",
                        "concordant",
                        f"Concordant positive finding: Visual symptoms indicate suspected Powdery Mildew, consistent with {pm_level} microclimatic favorability (optimal moderate temperatures and diurnal humidity swings).",
                        [
                            "Inspect flowering panicles and tender new flush leaves for white powdery conidial growth.",
                            "Ensure adequate orchard ventilation and consult an agronomist.",
                        ],
                    )
                else:
                    return (
                        "completed",
                        "discordant",
                        f"Discordant finding: Visual symptoms suggest Powdery Mildew, but sensor records show low favorability (suppressed by weather extremes or heavy rainfall).",
                        [
                            "Conduct ground verification to distinguish active powdery mildew from pesticide residue or dust.",
                        ],
                    )

            # Check Healthy leaf concordance
            elif "healthy" in clean_label:
                if anth_level == "high" or pm_level == "high":
                    elevated = "Anthracnose" if anth_level == "high" else "Powdery Mildew"
                    return (
                        "completed",
                        "discordant",
                        f"Discordant preventive finding: Sampled leaf is visually classified as healthy, but microclimatic conditions are highly favorable for {elevated}. Latent infection may develop if high moisture persists.",
                        [
                            "Scout the orchard over the next 3-7 days for emerging foliar lesions.",
                            "Maintain preventive monitoring of vulnerable flush leaves and panicles.",
                        ],
                    )
                else:
                    return (
                        "completed",
                        "concordant",
                        "Concordant healthy finding: Visual classification shows healthy leaf tissue, and sensor history demonstrates low disease favorability.",
                        [
                            "Continue standard orchard scouting and regular monitoring.",
                        ],
                    )

            # Other conditions / generic match
            return (
                "completed",
                "concordant",
                f"Multi-source assessment completed for {crop}. Visual classification: {img_label}; Sensor risk status: assessed.",
                [
                    "Conduct routine orchard inspection.",
                ],
            )

        # Fallback partial evidence
        return (
            "partial_evidence",
            "single_source",
            f"Partial assessment completed for {crop}. Image status: {img_status}; Sensor status: {sns_status}.",
            [
                "Collect supplementary visual and sensor observations.",
            ],
        )
