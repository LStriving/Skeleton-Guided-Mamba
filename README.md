# Skeleton-Guided-Mamba

Official code for the paper "Temporal Micro-action Localization with Skeleton-Guided Mamba for Videofluoroscopic Swallowing Study".

## 📢 News
- [2025/05/15] The supplementary material is available at [supplementary](./assets/Supplementary.pdf).
- [2025/05/13] 🔄The repository is created.

## Abstract

Videofluoroscopic Swallowing Study (VFSS) is the gold standard for assessing swallowing disorders,
enabling detailed analysis of swallowing phases.
Temporal micro-action localization in VFSS, which needs to identify and localize micro-actions
(e.g., hyoid motion, {'<'}2s), is critical for diagnosis but faces significant challenges: 1) **Spatial ambiguity**. Subtle anatomical movements are obscured by noise and blurred contours in X-ray images. 2) **Temporal complexity**. Micro-actions are extremely short, making them difficult to localize in lengthy videos.
Existing methods detect the whole swallowing before localizing micro-actions or trim videos to handle brief actions.
However, they fail to focus on key anatomical structures and struggle with efficient spatiotemporal modeling.
To address these issues, we propose Channel-enhanced Cross-Mamba (CCM), a framework that integrates 1) skeleton heatmap sequences to suppress noise and enhance key anatomical focus, and 2) a Mamba-based architecture with Channel-enhanced Cross-Mamba(CCM) to fuse the appearance features with skeleton guidance, enabling rich spatiotemporal features through efficient bidirectional modeling. Our framework achieves state-of-the-art performance, surpassing the previous method by 14.4\% in average mAP.

## Installation

Please refer to [INSTALL.md](./INSTALL.md) for installation.

## Performance

Comparison with State-of-the-Art Methods.
The best and second performances are highlighted in **bold** and _underline_, respectively.
"Ske." denotes skeleton input. Temporal action localization methods are applied _as-is_ to VFSS for temporal micro-action localization.
"Oracle" means we use the ground-truth of coarse stage as the input proposals to explore the upper bound of our method.

| Method            | Detector     | Ske. | 0.1      | 0.2      | 0.3      | 0.4      | 0.5      | 0.6      | 0.7      | Avg.     |
| ----------------- | ------------ | ---- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| A2Net             | A2Net        |      | 53.7     | 51.5     | 45.7     | 36.0     | 22.4     | 10.6     | 3.5      | 31.9     |
| ActionFormer      | ActionFormer |      | 76.8     | 74.4     | 69.7     | 59.3     | 48.8     | 38.5     | 24.6     | 56.0     |
| TriDet            | TriDet       |      | 79.6     | 76.6     | 72.4     | 63.7     | 53.1     | 40.9     | 26.2     | 58.9     |
| AdaTAD            | AdaTAD       |      | 81.0     | 77.4     | 70.0     | 62.3     | 54.4     | 42.1     | 24.8     | 58.9     |
| ActionMamba       | ActionMamba  |      | 82.0     | **79.3** | 74.3     | 67.4     | 53.1     | 37.1     | 19.9     | 59.0     |
| Ruan et al.       | A2Net        |      | 70.9     | 67.5     | 62.5     | 55.0     | 46.1     | 31.6     | 15.8     | 49.9     |
| Ruan et al.       | ActionMamba  |      | 77.9     | 75.0     | 69.5     | 62.2     | 54.8     | 45.2     | 28.9     | 59.1     |
| Hyder et al.      | ActionMamba  | ✓    | 76.8     | 74.6     | 69.4     | 62.0     | 56.2     | 45.2     | 30.8     | 59.3     |
| SG-Mamba (Ours)   | ActionMamba  | ✓    | _83.1_   | 78.5     | _74.6_   | _67.9_   | _59.0_   | _50.0_   | _37.2_   | _64.3_   |
| SG-Mamba (Oracle) | ActionMamba  | ✓    | **91.2** | **86.3** | **83.7** | **75.7** | **66.9** | **56.7** | **42.2** | **71.8** |


## Data Preparation

coming soon...

## Checkpoints
coming soon...

## Training

coming soon...

## Evaluation

coming soon...


