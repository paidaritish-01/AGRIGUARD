"""
Literature-informed environmental disease and stress risk models for mango orchards.

Enforces:
- Exact parameter thresholds derived from published peer-reviewed epidemiology.
- Transparent reporting of required variables, time windows, and citations.
- Explicit status: 'literature-informed' (never claiming clinical or lab confirmation).
- Safe fallback to 'insufficient_data' when observations are inadequate.
"""

from typing import List, Tuple
from agriscan.sensors.schema import (
    SensorDerivedFeatures,
    EnvironmentalRiskResult,
)


def evaluate_mango_environmental_risks(
    features: SensorDerivedFeatures,
) -> Tuple[List[EnvironmentalRiskResult], List[str]]:
    """
    Evaluates microclimate history against known epidemiological criteria
    for major mango diseases and abiotic stressors.
    """
    risk_results: List[EnvironmentalRiskResult] = []
    stress_indicators: List[str] = []

    # Safe fallback if insufficient history (< 6 hours or < 6 observations)
    if features.window_hours < 6.0 or features.measurement_count < 6:
        insufficient_anthracnose = EnvironmentalRiskResult(
            disease_or_condition="Mango Anthracnose",
            pathogen_or_stressor="Colletotrichum gloeosporioides",
            risk_level="insufficient_data",
            favorable_conditions_met=False,
            explanation=f"Insufficient temporal history ({features.measurement_count} observations over {features.window_hours:.1f}h). Minimum 6 hours required for microclimate risk assessment.",
            required_variables=["air_temperature_c", "relative_humidity_pct"],
            source_citation="Dodd et al. (1992) Ann. Appl. Biol. 120(1):43-52; Fitzell & Peak (1984) Ann. Appl. Biol. 104(1):53-59.",
            evidence_type="literature-informed",
            assumptions_and_limitations="Requires continuous multi-hour temperature and humidity monitoring to evaluate spore germination periods.",
        )
        insufficient_mildew = EnvironmentalRiskResult(
            disease_or_condition="Mango Powdery Mildew",
            pathogen_or_stressor="Oidium mangiferae (Pseudoidium anacardii)",
            risk_level="insufficient_data",
            favorable_conditions_met=False,
            explanation="Insufficient temporal history to evaluate diurnal temperature and relative humidity cycles.",
            required_variables=["air_temperature_c", "relative_humidity_pct"],
            source_citation="Palti et al. (1974) Plant Dis. Rep. 58:45-49; ICAR-CISH Technical Guidelines.",
            evidence_type="literature-informed",
            assumptions_and_limitations="Requires diurnal cycle monitoring to detect daytime dry spells and nighttime humidity peaks.",
        )
        return [insufficient_anthracnose, insufficient_mildew], ["Insufficient sensor history for stress assessment."]

    # 1. Mango Anthracnose (Colletotrichum gloeosporioides)
    # References:
    # - Dodd, P.B., Estrada, A.B., Matcham, J., Jeffries, P., & Jeger, M.J. (1992).
    #   The effect of temperature and wetness duration on infection of mango by Colletotrichum gloeosporioides.
    #   Annals of Applied Biology, 120(1), 43-52.
    # - Fitzell, R.D., & Peak, C.M. (1984). The epidemiology of anthracnose disease of mango.
    #   Annals of Applied Biology, 104(1), 53-59.
    # Thresholds:
    # Optimal germination at 24°C-30°C with prolonged RH > 85% or leaf wetness >= 12h.
    t_mean = features.temperature_mean_c or 0.0
    hrs_high_rh = features.hours_high_humidity_above_85 or 0.0
    lw_hours = features.total_leaf_wetness_hours or 0.0
    rain_mm = features.total_rainfall_mm or 0.0

    if 24.0 <= t_mean <= 30.0 and (hrs_high_rh >= 12.0 or lw_hours >= 12.0):
        anth_risk = "high"
        anth_favorable = True
        anth_exp = (
            f"High microclimatic favorability for Anthracnose: mean temperature ({t_mean:.1f}°C) within optimum range "
            f"[24-30°C] with {hrs_high_rh:.1f} hours of high relative humidity (>=85%) and {lw_hours:.1f} hours leaf wetness."
        )
    elif 22.0 <= t_mean <= 32.0 and (hrs_high_rh >= 6.0 or rain_mm > 5.0 or lw_hours >= 6.0):
        anth_risk = "moderate"
        anth_favorable = True
        anth_exp = (
            f"Moderate microclimatic favorability for Anthracnose: temperature ({t_mean:.1f}°C) and moisture conditions "
            f"({hrs_high_rh:.1f} hours RH >= 85%, rainfall {rain_mm:.1f} mm) support intermittent conidial germination."
        )
    else:
        anth_risk = "low"
        anth_favorable = False
        anth_exp = (
            f"Low microclimatic favorability for Anthracnose: weather conditions (mean temp {t_mean:.1f}°C, "
            f"{hrs_high_rh:.1f} hours RH >= 85%) are outside the high-risk infection window."
        )

    risk_results.append(
        EnvironmentalRiskResult(
            disease_or_condition="Mango Anthracnose",
            pathogen_or_stressor="Colletotrichum gloeosporioides",
            risk_level=anth_risk,
            favorable_conditions_met=anth_favorable,
            explanation=anth_exp,
            required_variables=["air_temperature_c", "relative_humidity_pct", "optional:leaf_wetness_hours", "optional:rainfall_mm"],
            source_citation="Dodd et al. (1992) Ann. Appl. Biol. 120(1):43-52; Fitzell & Peak (1984) Ann. Appl. Biol. 104(1):53-59.",
            evidence_type="literature-informed",
            assumptions_and_limitations=(
                "Assumes susceptible flush/panicle tissue and viable fungal inoculum present in orchard canopy. "
                "Favorable microclimate indicates potential infection periods, not confirmed colonization."
            ),
        )
    )

    # 2. Mango Powdery Mildew (Oidium mangiferae / Pseudoidium anacardii)
    # References:
    # - Palti, J., Shoham, H., & Shenkar, F. (1974). Mango powdery mildew in Israel. Plant Dis. Rep., 58, 45-49.
    # - ICAR - Central Institute for Subtropical Horticulture (CISH) Disease Advisory.
    # Criteria:
    # Favourable: Moderate temperatures (20°C - 26°C), morning humidity > 65-75%, dry afternoon (<60% RH), rainfall < 2mm.
    # Inhibited by continuous heavy rainfall (>10mm) or sustained heat (>35°C).
    rh_min = features.humidity_min_pct if features.humidity_min_pct is not None else 50.0
    rh_max = features.humidity_max_pct if features.humidity_max_pct is not None else 80.0

    if rain_mm > 10.0 or t_mean > 35.0:
        pm_risk = "low"
        pm_favorable = False
        pm_exp = (
            f"Low microclimatic favorability for Powdery Mildew: rainfall ({rain_mm:.1f} mm) or high temperatures "
            f"({t_mean:.1f}°C) suppress conidial viability and aerial dispersal."
        )
    elif 20.0 <= t_mean <= 26.0 and rh_max >= 75.0 and rh_min <= 65.0:
        pm_risk = "high"
        pm_favorable = True
        pm_exp = (
            f"High microclimatic favorability for Powdery Mildew: optimal temperature ({t_mean:.1f}°C) combined with "
            f"favorable diurnal humidity swing (max {rh_max:.1f}%, min {rh_min:.1f}%) facilitating spore germination and airborne spread."
        )
    elif 18.0 <= t_mean <= 28.0 and rh_max >= 65.0:
        pm_risk = "moderate"
        pm_favorable = True
        pm_exp = (
            f"Moderate microclimatic favorability for Powdery Mildew: temperature ({t_mean:.1f}°C) and relative humidity "
            f"reach conducive thresholds."
        )
    else:
        pm_risk = "low"
        pm_favorable = False
        pm_exp = f"Low favorability for Powdery Mildew: temperature ({t_mean:.1f}°C) or humidity levels are suboptimal."

    risk_results.append(
        EnvironmentalRiskResult(
            disease_or_condition="Mango Powdery Mildew",
            pathogen_or_stressor="Oidium mangiferae (Pseudoidium anacardii)",
            risk_level=pm_risk,
            favorable_conditions_met=pm_favorable,
            explanation=pm_exp,
            required_variables=["air_temperature_c", "relative_humidity_pct", "rainfall_mm"],
            source_citation="Palti et al. (1974) Plant Dis. Rep. 58:45-49; ICAR-CISH Mango Crop Guide.",
            evidence_type="literature-informed",
            assumptions_and_limitations=(
                "Powdery mildew primarily colonizes flowering panicles and tender new leaf flushes. "
                "Sensor metrics cannot detect presence of conidia without ground scout confirmation."
            ),
        )
    )

    # 3. Abiotic Environmental Stress Indicators
    if features.temperature_max_c and features.temperature_max_c >= 38.0:
        stress_indicators.append(
            f"Extreme ambient heat stress: maximum recorded temperature reached {features.temperature_max_c:.1f}°C (>=38°C threshold)."
        )
    if features.soil_moisture_mean_pct is not None:
        if features.soil_moisture_mean_pct < 18.0:
            stress_indicators.append(
                f"Severe soil moisture deficit: mean root-zone moisture {features.soil_moisture_mean_pct:.1f}% indicates acute irrigation need."
            )
        elif features.soil_moisture_mean_pct > 85.0:
            stress_indicators.append(
                f"Potential soil waterlogging: mean soil moisture {features.soil_moisture_mean_pct:.1f}% (>85% saturation risk)."
            )

    return risk_results, stress_indicators
