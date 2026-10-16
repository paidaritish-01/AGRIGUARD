# AgriScan Research References & Agricultural Guidance

This document consolidates the scientific foundation, model choices, dataset specifications, and epidemiological microclimate risk rules used in AgriScan.

---

## 1. Vision Model Architecture: EfficientNet-B0

### Reference
- **Paper**: Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. International Conference on Machine Learning (ICML 2019), PMLR 97:6105-6114.
- **Link**: [https://proceedings.mlr.press/v97/tan19a.html](https://proceedings.mlr.press/v97/tan19a.html)
- **PyTorch Documentation**: [https://docs.pytorch.org/vision/stable/models/efficientnet.html](https://docs.pytorch.org/vision/stable/models/efficientnet.html)

### Selection Rationale
- **Parameter Efficiency**: EfficientNet-B0 utilizes compound coefficient scaling with inverted bottleneck residual blocks (MBConv) and Squeeze-and-Excitation optimization.
- **Footprint**: ~5.3 million parameters compared to ~11.7 million in ResNet18, achieving higher top-1 accuracy on ImageNet with ~0.39 billion FLOPs.
- **Suitability for AgriScan**: Practical for credit-efficient, low-latency deployment and edge drone/farm workstation evaluation.
- **Architecture Flexibility**: AgriScan's backbone factory (`agriscan/models/backbone.py`) supports both `efficientnet_b0` and `resnet18` via torchvision, enabling seamless transfer learning without proprietary dependencies.

---

## 2. Mango Leaf Dataset: SAR-MLD1-2025

### References
- **Paper**: *SAR-MLD1-2025: A comprehensive dataset of mango leaf diseases in Bangladesh*. Data in Brief, 2026. [https://doi.org/10.1016/j.dib.2026.112826](https://doi.org/10.1016/j.dib.2026.112826)
- **Dataset Part 1**: Mendeley Data, [https://data.mendeley.com/datasets/sd8hzpg69b/5](https://data.mendeley.com/datasets/sd8hzpg69b/5)
- **Dataset Part 2**: Mendeley Data, [https://data.mendeley.com/datasets/j3bn63t4sp/5](https://data.mendeley.com/datasets/j3bn63t4sp/5)
- **Licence**: Creative Commons Attribution 4.0 International (CC BY 4.0).

### Dataset Specifications
- **Reported Volume**: 4,921 images partitioned across two published parts.
- **Classes (5)**:
  1. `healthy`
  2. `anthracnose` (*Colletotrichum gloeosporioides*)
  3. `powdery_mildew` (*Oidium mangiferae*)
  4. `turning_brown` (physiological senescence or nutritional scorch)
  5. `gall_midge` (*Procontarinia* spp. galling damage)
- **Collection Context**: Captured in Bangladesh across localized orchards. The methodology notes leaf samples photographed primarily against uniform or white backgrounds.

### Critical Limitations & Evaluation Safeguards
- **Close-up vs. Aerial Domain Gap**: SAR-MLD1-2025 is strictly a close-up leaf dataset. It CANNOT validate aerial drone canopy stress detection or fruit pathology.
- **Background Bias**: Studio/white-background leaf photography lacks natural orchard canopy clutter, sunlight variations, shadows, and soil reflections. Performance on field-captured drone imagery will degrade without in-domain fine-tuning.
- **Data Leakage Risk**: Augmented copies and multiple captures of the same leaf must be partitioned by tree or session ID. Random splitting across duplicated leaves results in over-optimistic evaluation metrics.
- **Credit Budget Compliance**: In accordance with the budget policy, the dataset is not automatically downloaded during initial code setup. A dataset validation workflow is provided in `scripts/validate_mango_dataset.py` and configured in `configs/mango_close_up_config.yaml`.

---

## 3. Environmental Epidemiological Risk Rules

The sensor module evaluates microclimatic conditions against peer-reviewed epidemiological criteria.
All rules are marked as **literature-informed** (not locally calibrated or laboratory confirmed).

### Rule 1: Mango Anthracnose (*Colletotrichum gloeosporioides*)
- **Pathogen**: *Colletotrichum gloeosporioides* (Penz.) Penz. & Sacc.
- **Epidemiological Sources**:
  1. *Dodd, P. B., Estrada, A. B., Matcham, J., Jeffries, P., & Jeger, M. J. (1992)*. The effect of temperature and wetness duration on infection of mango by *Colletotrichum gloeosporioides*. *Annals of Applied Biology*, 120(1), 43-52.
  2. *Fitzell, R. D., & Peak, C. M. (1984)*. The epidemiology of anthracnose disease of mango: inoculum sources, spore production and infection periods. *Annals of Applied Biology*, 104(1), 53-59.
  3. *University of Florida IFAS Extension (EDIS PP-38)*: Anthracnose on Mango.
- **Required Variables**: `air_temperature_c`, `relative_humidity_pct`, optional `leaf_wetness_hours`, optional `rainfall_mm`.
- **Time Window**: Preceding 24 to 72 hours.
- **Exact Thresholds**:
  - **Optimal infection range**: Conidial germination and appressorium formation peak between **24.0°C and 30.0°C** (thermal optimum ~25°C–28°C).
  - **High Risk**: Mean temperature within [24.0°C, 30.0°C] AND (hours with RH >= 85% is >= 12h OR cumulative leaf wetness >= 12h).
  - **Moderate Risk**: Mean temperature within [22.0°C, 32.0°C] AND (hours with RH >= 85% is >= 6h OR rainfall > 5mm).
  - **Low Risk**: Below thresholds or temperatures inhibitory (<18°C or >35°C).
- **Assumptions & Caveats**: Microclimate favorability indicates favorable conditions for spore germination, NOT confirmed leaf colonization or disease manifestation.

---

### Rule 2: Mango Powdery Mildew (*Oidium mangiferae* / *Pseudoidium anacardii*)
- **Pathogen**: *Oidium mangiferae* Berthet (syn. *Pseudoidium anacardii*).
- **Epidemiological Sources**:
  1. *Palti, J., Shoham, H., & Shenkar, F. (1974)*. Mango powdery mildew in Israel. *Plant Disease Reporter*, 58, 45-49.
  2. *ICAR-Central Institute for Subtropical Horticulture (CISH)*: Technical Guidelines for Mango Crop Protection.
- **Required Variables**: `air_temperature_c`, `relative_humidity_pct`, optional `rainfall_mm`.
- **Time Window**: Preceding 24 to 72 hours.
- **Exact Thresholds**:
  - **Thermal optimum**: Moderate temperatures between **20.0°C and 26.0°C**.
  - **Humidity dynamics**: High night/morning humidity (>65%–75%) promotes conidial germination, while dry, warm daytime conditions (<60% RH) accelerate airborne conidial detachment and dispersal.
  - **Inhibition**: Heavy rainfall (>10 mm) or continuous standing leaf wetness washes off or bursts conidia; temperatures >35°C suppress viability.
  - **High Risk**: Mean temperature within [20.0°C, 26.0°C] AND diurnal RH swing exists (maximum RH >= 75% AND minimum RH <= 65%) AND rainfall <= 2.0 mm.
  - **Moderate Risk**: Mean temperature within [18.0°C, 28.0°C] AND maximum RH >= 65% AND rainfall <= 5.0 mm.
  - **Low Risk**: Outside conducive ranges or heavy rainfall event (>10 mm).
- **Assumptions & Caveats**: Fungal conidia primarily attack emergent flushes and flowering inflorescences. Sensor data alone cannot confirm conidia presence.

---

### Rule 3: Abiotic Environmental Stress Indicators
- **Sources**: FAO Irrigation and Drainage Paper 56; University of Florida IFAS Mango Production Guide.
- **Variables**: `air_temperature_c`, `soil_moisture_pct`.
- **Thresholds**:
  - **Heat Stress**: Maximum temperature >= **38.0°C**.
  - **Soil Moisture Deficit (Drought)**: Mean root-zone soil moisture < **18.0%**.
  - **Waterlogging Risk**: Mean root-zone soil moisture > **85.0%**.

---

## 4. Multi-Source Assessment Synthesis Logic

The combined assessment coordinator (`agriscan/assessment/combined.py`) evaluates independent evidence streams without mathematical conflation or ungrounded probability multiplication:

| Image Finding | Sensor Risk Finding | Evidence Concordance | Combined Action / Explanation |
| :--- | :--- | :--- | :--- |
| **Suspected Anthracnose** | High / Moderate Anthracnose | **Concordant** | High confidence multi-evidence alert. Scout canopy for active foliar lesions. |
| **Suspected Anthracnose** | Low Anthracnose Risk | **Discordant** | Lesions may represent historical damage from earlier rainfall, or abiotic leaf scorch. Verify lesion activity. |
| **Healthy Leaf** | High Anthracnose Risk | **Discordant** | Preventive warning: Leaf is currently clean, but microclimate is highly favorable. Monitor flush leaves over next 3-7 days. |
| **Healthy Leaf** | Low Risk | **Concordant** | Low disease pressure. Maintain routine orchard scouting. |
| **Model Unavailable** | Assessed Risk | **Single Source** | Vision model unavailable; assessment based solely on microclimate risk. |
| **Visual Classification** | Not Assessed | **Single Source** | Assessment based solely on image; sensor history absent. |
| **Model Unavailable** | Insufficient Data | **Insufficient** | Neither stream available. Return insufficient data status. |
