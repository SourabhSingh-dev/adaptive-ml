# AdaptiveML

### Adaptive Model Routing & Uncertainty-Aware Decision Engine

AdaptiveML investigates whether **different ML models are better suited to different inputs**, and whether a routing layer can dynamically select the appropriate model instead of relying on a single global model.

The project uses the **HHAR (Heterogeneity Human Activity Recognition)** dataset and evaluates a heterogeneous pool of classical and deep learning models.

---

## Problem

A conventional ML system looks like:

```text
Input → Best Global Model → Prediction
```

But a model that performs best **on average** may not be the best model for every individual sample.

AdaptiveML investigates:

```text
Input
  ↓
Candidate Models
  ↓
Which model should handle this sample?
  ↓
Selected Model
  ↓
Prediction + Reliability
```

The system is also designed around a practical **accuracy–cost trade-off**: using a more capable model may improve prediction quality but can increase inference cost. The long-term goal is therefore to select the appropriate model based on both **sample difficulty and inference cost**, rather than maximizing accuracy alone.

---

## Model Pool

| Model               | Type          |   Accuracy |
| ------------------- | ------------- | ---------: |
| Logistic Regression | Classical     |     71.61% |
| Random Forest       | Classical     |     74.55% |
| XGBoost             | Classical     |     74.95% |
| 1D CNN              | Deep Learning |     76.18% |
| LSTM                | Deep Learning | **78.27%** |

---

## Key Finding

Evaluation was performed on **39,596 samples**.

| Analysis                |        Result |
| ----------------------- | ------------: |
| Best individual model   | LSTM — 78.27% |
| Oracle routing accuracy |    **98.33%** |
| All models correct      |         9,864 |
| All models wrong        |           663 |
| Model disagreement      |        29,069 |

The **98.33% oracle accuracy is not actual router performance**. It represents the theoretical upper bound obtained if the system knew which candidate model would be correct for each sample.

The important observation is that the models disagree substantially, creating a potential opportunity for learned model routing.

---

## Accuracy vs. Cost

AdaptiveML does not assume that the most accurate model should always be used.

Conceptually:

```text
                 Accuracy
                    ↑
                    │
              LSTM  │
               CNN  │
             XGBoost│
                RF  │
                LR  │
                    └──────────────→ Inference Cost
```

The eventual routing system will investigate whether it is possible to achieve a better **accuracy–latency–compute trade-off** by dynamically choosing models.

For example:

```text
Easy sample       → Cheaper model
Difficult sample  → More capable model
Uncertain sample  → Review / Abstain
```

Actual latency and compute benchmarks will be added as the routing system develops.

---

## Architecture

### Current

```text
HHAR Sensor Data
       ↓
Preprocessing / Windowing
       ↓
┌────────┬────────┬────────┬────────┬────────┐
│   LR   │   RF   │ XGBoost│  CNN   │  LSTM  │
└────────┴────────┴────────┴────────┴────────┘
       ↓
Per-Sample Predictions
       ↓
Error & Disagreement Analysis
```

### Target

```text
Input
  ↓
Candidate Models
  ↓
Routing Layer
  ↓
Selected Model
  ↓
Prediction
  ↓
Confidence / Uncertainty
  ↓
AUTO / REVIEW / ABSTAIN
```

---

## Dataset

**HHAR — Heterogeneity Human Activity Recognition**

The dataset contains real-world smartphone sensor measurements across different users and devices for six activities:

`bike` · `sit` · `stand` · `walk` · `stairsup` · `stairsdown`

The raw dataset is not included in this repository.

See [`DATA_CARD.md`](DATA_CARD.md) for dataset details and analysis.

---

## Current Status

### Completed

* Dataset investigation and preprocessing
* Classical ML baselines
* 1D CNN and LSTM models
* Per-sample prediction generation
* Error matrix analysis
* Model disagreement analysis
* Oracle routing analysis

### Next

* Leakage-safe learned router
* Routing regret evaluation
* Accuracy–latency benchmarking
* Probability calibration
* Uncertainty estimation
* Selective prediction / abstention
* Cross-user and cross-device evaluation
* Distribution-shift analysis
* Deployment

---

## Tech Stack

**Python · Polars · NumPy · Scikit-learn · XGBoost · PyTorch**

---

## Run

```bash
git clone <repository-url>
cd adaptive_ml

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Run the current model-pool analysis:

```bash
cd src
python analyze_full_pool.py
```

---

## Research Direction

The core question of AdaptiveML is:

> **Instead of asking "Which model is best?", can we learn "Which model should handle this sample?"**

The project will evaluate whether adaptive routing can convert model diversity into improved **accuracy, efficiency, and decision reliability**.
