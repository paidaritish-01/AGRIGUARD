# AgriScan Model Card

## Model Overview
- **Model Name:** AgriScan Dual-Mode Crop Health Diagnostic System (CropSentra / Mango AI)
- **Model Version:** 1.0.0
- **Model Type:** Deep Convolutional Neural Network (Transfer Learning via EfficientNet-B0 / ResNet-18) & Literature-Informed Environmental Microclimate Risk Engine
- **Framework:** PyTorch 2.x / Torchvision / FastAPI
- **Primary Crop:** Mango (*Mangifera indica*)
- **Additional Supported Crops:** Tomato (*Solanum lycopersicum*), Corn, Wheat, Apple, General Canopy
- **Release Date:** October 2026
- **License:** Apache 2.0 (Code) / Dataset licensing subject to underlying repositories (SAR-MLD1-2025: CC BY 4.0)

---

## Intended Use & Analysis Modes

AgriScan is designed exclusively as an agronomic decision-support and screening tool. It implements **two strictly separated analysis modes**:

1. **Close-Up Classification Mode (`close_up`):**
   - **Target Subject:** High-resolution photographs of individual leaves, stems, or fruits taken from distances under 1 meter.
   - **Output:** Categorization into supported plant pathology classes (e.g., `tomato_healthy`, `tomato_early_blight`, `tomato_late_blight`, `tomato_bacterial_spot`).
   - **Decision Support:** Provides preliminary triage for scouted field plants.

2. **Aerial Analysis Mode (`aerial`):**
   - **Target Subject:** Geotagged high-resolution drone orthomosaics or nadir aerial flight captures (flight altitudes 15m–60m).
   - **Output:** Canopy anomaly localization using tiled overlapping inference (`healthy_canopy`, `visible_crop_stress`, `abnormal_canopy_void`) with global pixel bounding boxes.
   - **Decision Support:** Identifies spatial stress zones across an entire field for prioritized agronomist scouting.

---

## Critical Agricultural & Scientific Limitations

> [!WARNING]
> **Domain Disclaimers & Operational Boundaries**

1. **Non-Equivalence of Close-up and Aerial Modalities:**
   A model trained on close-up leaf photographs **cannot and must not** be treated as validated for aerial drone imagery. Canopy-level spectral reflection, scale, shadow occlusion, and atmospheric scattering differ fundamentally from macro leaf photographs.
2. **Mango Leaf vs Fruit/Orchard Detection (SAR-MLD1-2025):**
   The reference mango dataset (SAR-MLD1-2025) contains isolated leaves photographed against white or uniform backgrounds. White-background classification **cannot guarantee performance on natural field foliage**, nor does it validate detection of fruit rot or canopy-level infestations.
3. **Environmental Sensor Risk vs Clinical Diagnosis:**
   Microclimatic sensor analysis evaluates epidemiological favorability (temperature, humidity, leaf wetness conducive to fungal spore germination). It **does not confirm biological pathogen presence, infection, or colonization**. High favorability warrants orchard scouting, not automatic chemical application.
4. **No Physical Sensors / Hardware Ingestion Status:**
   Current sensor history operations utilize documented CSV schemas and synthetic demonstration datasets. Physical telemetry integrations must adhere to the versioned CSV contract.
5. **Ambiguity of RGB Crop Stress:**
   Standard RGB imagery cannot reliably discriminate fungal or bacterial disease from physiological water stress, nitrogen deficiency, soil compaction, salinity, or insect herbivory without confirmatory laboratory assays or in-field agronomist ground-truthing.
6. **Drone GPS vs Plant Coordinates:**
   Metadata extracted from image EXIF records the GPS location and altitude of the drone aircraft / gimbal sensor at exposure time. It **does not** infer the exact coordinates of an individual plant due to gimbal tilt angle, altitude, camera FOV, and field topography.
7. **No Chemical Intervention Recommendations:**
   AgriScan strictly outputs diagnostic triage observations (e.g. "suspected Anthracnose" or "visible crop stress"). It **does not** generate pesticide, fungicide, or chemical recommendations under any circumstances.
8. **Out-of-Distribution & Unsupported Crops:**
   Confidence thresholds mitigate low-confidence in-distribution classification errors, but **cannot reliably detect every unsupported crop or out-of-distribution image**. Images of non-crop objects or unsupported species may produce arbitrary activations.

---

## Dataset Pipeline & Verification Contract

### Group-Based Leakage Prevention
To ensure robust generalization and prevent data leakage:
- Images are grouped by `farm_id` or `session_id`.
- Splits (Train: 70%, Validation: 15%, Test: 15%) are partitioned by group, guaranteeing that all images from a specific farm or flight flight session reside exclusively in a single split.

### Dataset Integrity Checks
- **Readability & Decode:** Rejection of 0-byte files, truncated JPEG streams, or invalid headers.
- **Duplicate Detection:** Pre-training SHA-256 content hashing across all directories and splits.
- **Class Imbalance Handling:** Automated computation of inverse-frequency class weights for the CrossEntropy loss function.

### Real Dataset Sourcing & Licensing Guidelines
To train production-grade models:
- **Licensing:** Only integrate datasets with explicit commercial or research permissions (e.g., CC-BY-4.0, MIT, or institutional field-trial data sharing agreements).
- **Public Reference Benchmarks:** PlantVillage (macro leaf photographs, CC-BY), PlantDoc, or specialized aerial drone benchmarks (e.g., Agriculture-Vision, UAVid).
- **Adding Custom Field Data:** Structure images into `data/<mode>/<class_name>/` or populate `metadata.csv` with `image_path,label,crop,farm_id,session_id,gps_lat,gps_lon`.

---

## Evaluation Methodology & Metrics

1. **Class Triage Metrics:**
   - Per-class Precision, Recall, and F1-score.
   - Macro-averaged F1 to prevent majority class bias.
   - Multi-class Confusion Matrix.
2. **Generalization Metrics:**
   - Performance segmented by held-out farm ID to verify domain transfer.
3. **Uncertainty & Calibration:**
   - **Abstention Threshold:** Tuned strictly on the validation split (never the test set) to achieve target diagnostic accuracy. Predictions below the threshold return `status = "uncertain"`.
   - **Expected Calibration Error (ECE):** Evaluated over 10 confidence bins. Raw softmax scores are labeled as `raw_softmax_confidence` unless temperature scaling is fitted.

---

## Hardware Requirements

- **Production Training:** 1x NVIDIA GPU with >= 8GB VRAM (e.g., RTX 3070, T4, or A10) recommended for full batch transfer learning.
- **Inference & Smoke Tests:** 2+ CPU cores, 4GB RAM. CPU execution is fully supported for edge drone gateways and API servers.
