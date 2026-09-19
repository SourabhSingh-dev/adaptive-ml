# AdaptiveML

## Adaptive Model Routing & Uncertainty-Aware Decision Engine

AdaptiveML investigates whether **different machine-learning models are better suited to different inputs**, and whether a routing layer can exploit those differences instead of relying on one fixed model for every sample.

The project is built around a simple question:

> **Can we dynamically select the model that is better suited for each individual input while considering prediction performance and inference cost?**

The system uses multiple independent candidate models, analyzes their per-sample behavior, and trains a learned router to select a model for a given sensor window.

---

## Project Objective

AdaptiveML is **not** trying to prove that one model family is universally better than another.

The objective is to investigate whether:

> **Different models are better for different inputs, and an intelligent inference layer can exploit that difference.**

This leads to a model-routing architecture:

```text
Sensor Window
      │
      ▼
Feature Extraction
      │
      ▼
   Model Router
      │
      ├──────────────┐
      ▼              ▼
Selected Model   Model Pool
      │
      ▼
Prediction
```

The candidate models remain independent. The router acts as the decision layer that determines which candidate should handle a particular sample.

---

# Dataset

AdaptiveML uses the **Heterogeneity Human Activity Recognition (HHAR)** dataset.

The project works with smartphone and smartwatch sensor data containing activities such as:

* `bike`
* `sit`
* `stairsdown`
* `stairsup`
* `stand`
* `walk`

The current implementation uses accelerometer data from:

```text
Phones_accelerometer.csv
Phones_gyroscope.csv
Watch_accelerometer.csv
Watch_gyroscope.csv
```

The current training pipeline specifically loads:

```text
data/raw/Phones_accelerometer.csv
```

---

# Data Processing

The raw accelerometer stream is converted into overlapping sensor windows.

The current preprocessing implementation uses:

* window size: **128 observations**
* step size: **64 observations**
* three sensor channels: `x`, `y`, `z`

Each window becomes one prediction sample.

The preprocessing code also groups the raw data by:

```text
User
Device
Activity
```

before constructing windows.

Two representations are then used:

### Sequential representation

The original window is retained as a 3D tensor:

```text
(samples, time, channels)
```

This representation is used by the deep-learning models.

### Tabular representation

The temporal dimension is summarized using:

* mean
* standard deviation
* maximum
* minimum

for each of the three sensor channels.

This produces the tabular representation used by the classical models and the router.

---

# Group-Aware Evaluation

The project deliberately uses a **GroupShuffleSplit based on users** rather than randomly mixing observations from the same user between training and testing.

The current base-model split is:

```text
Test size: 20%
random_state: 42
group: User
```

The deep-learning training pipeline recreates the same group-aware split.

The router also performs its own group-aware split on the valid routing samples.

This is important because the objective is to evaluate model behavior on unseen users rather than allowing observations from the same user to appear in both training and testing.

---

# Candidate Model Pool

AdaptiveML currently contains five candidate models.

| Model               | Representation | Role                    |
| ------------------- | -------------- | ----------------------- |
| Logistic Regression | Tabular        | Linear baseline         |
| Random Forest       | Tabular        | Tree ensemble           |
| XGBoost             | Tabular        | Gradient boosting       |
| 1D CNN              | Raw sequence   | Local temporal patterns |
| LSTM                | Raw sequence   | Sequential patterns     |

The classical models operate on engineered tabular features.

The CNN and LSTM operate directly on the sensor sequence.

### 1D CNN

The current CNN contains three convolutional stages:

```text
3 → 64 → 128 → 256
```

followed by adaptive average pooling and a classifier.

### LSTM

The current LSTM uses:

```text
input_size = 3
hidden_size = 64
num_layers = 1
```

and uses the final sequence representation for classification.

---

# Why Independent Models?

The candidate models are intentionally kept independent.

AdaptiveML is trying to determine whether the models have **different strengths and different failure patterns**.

Passing the mistakes of one candidate into another would turn the architecture into a different type of system, such as stacking or boosting.

XGBoost itself uses sequential error correction internally between its trees, but that is different from the architecture being investigated here.

AdaptiveML instead follows:

```text
Independent Candidate Models
            │
            ▼
       Model Router
            │
            ▼
     Selected Candidate
```

This allows the project to analyze which model is actually better for individual samples.

---

# Per-Sample Model Analysis

A central part of AdaptiveML is that it does not only record aggregate model accuracy.

Each model stores:

```text
sample_id
true_label
model_name
prediction
confidence
```

These predictions are combined into an **error matrix**.

For every sample, the system records whether each candidate model was correct.

This allows the project to investigate:

* model disagreement
* complementary errors
* samples where every model succeeds
* samples where every model fails
* the potential benefit of routing

---

# Current Model-Pool Results

The current `analyze_full_pool.py` run produced:

| Model               | Accuracy |
| ------------------- | -------: |
| Logistic Regression |   71.61% |
| Random Forest       |   74.55% |
| XGBoost             |   74.95% |
| 1D CNN              |   76.18% |
| LSTM                |   78.27% |

The test set contains:

```text
39,596 samples
```

The error-matrix analysis produced:

```text
All Models Wrong:       663
All Models Right:      9,864
Disagreement:         29,069
```

The row-wise oracle accuracy is:

```text
98.33%
```

The oracle represents the accuracy obtained if, for every sample, we could always select a model that is correct.

It is therefore a measure of the **available routing potential in the current candidate pool**, not the performance of the actual router.

---

# Final System-Level Results

The final evaluation compares three inference strategies on the unseen test set:

| System                     |  Accuracy | Avg Latency Cost |
| -------------------------- | --------: | ---------------: |
| Always Logistic Regression | **71.6%** |        **1.00×** |
| Always LSTM                | **78.3%** |        **3.37×** |
| **AdaptiveML Router**      | **81.5%** |        **1.85×** |

The comparison demonstrates the behavior that AdaptiveML is designed to investigate: rather than committing every sample to a single model, the router dynamically selects among the candidate models.

The reported AdaptiveML result is:

```text
Accuracy:          81.5%
Average Cost:      1.85×
```

For comparison:

```text
Always Logistic Regression
Accuracy:          71.6%
Average Cost:      1.00×

Always LSTM
Accuracy:          78.3%
Average Cost:      3.37×
```

These results are from the project's final system-level evaluation recorded in the development process.


# Why the Oracle Matters

Consider a sample where:

```text
Logistic Regression → wrong
Random Forest       → correct
XGBoost             → wrong
1D CNN              → wrong
LSTM                → correct
```

A static model must commit to one candidate.

An ideal router could select either Random Forest or LSTM for that particular sample.

The current error analysis shows that a large number of test samples contain disagreement between the candidate models.

That disagreement is the region where adaptive routing can potentially provide value.

The 663 samples where all five models are wrong also show an important limitation:

> Routing cannot fix an error when no candidate in the current model pool produces the correct prediction.

---

# Inference Cost

AdaptiveML does not treat model accuracy as the only consideration.

Different models have different inference costs.

The project therefore performs an empirical latency benchmark.

The benchmark:

1. loads each saved model;
2. performs 10 warm-up executions;
3. performs 100 measured executions;
4. calculates average milliseconds per prediction;
5. normalizes the latency relative to Logistic Regression.

The normalized cost map currently used by the router is:

```text
Logistic_Regression : 1.00
Random_Forest       : 123.84
XGBoost             : 2.41
1D_CNN              : 3.69
LSTM                : 3.37
```

The purpose is to give the router a cost signal rather than treating every candidate model as equally expensive.

---

# Cost-Aware Routing

The router does not define the target simply as:

```text
Which model is correct?
```

Instead, the current implementation constructs a utility:

```text
Utility = Correctness - COST_PENALTY × Model Cost
```

with:

```text
COST_PENALTY = 0.01
```

The model with the highest utility becomes the routing target whenever at least one candidate model is correct.

This makes the routing problem explicitly aware of inference cost.

---

# Learned Router

The current router is an **XGBoost classifier**.

Its input is the tabular representation extracted from the sensor window.

Its target is the model selected by the cost-aware utility calculation.

The router uses:

```text
GroupShuffleSplit
test_size = 0.3
random_state = 42
```

for the router-level split.

Balanced sample weights are also calculated for router training.

The trained artifacts are saved as:

```text
models/router_xgb.pkl
models/router_label_encoder.pkl
```

---

# Adaptive Inference Pipeline

The current inference pipeline loads:

* Logistic Regression
* Random Forest
* XGBoost
* 1D CNN
* LSTM
* trained router
* router label encoder

For a new sensor window:

```text
Raw Sensor Window
        │
        ▼
Tabular Feature Extraction
        │
        ▼
      Router
        │
        ▼
 Selected Model
        │
        ▼
Activity Prediction
```

The pipeline returns:

```python
{
    "routed_to": selected_model_name,
    "predicted_activity": activity
}
```

The current activity mapping is:

```text
0 → bike
1 → sit
2 → stairsdown
3 → stairsup
4 → stand
5 → walk
```

---

# System Evaluation

The project contains `evaluate_system.py` for comparing:

```text
Always Logistic Regression
Always LSTM
AdaptiveML Router
```

The comparison includes:

* predictive accuracy
* average normalized latency cost

The broader project discussion also identifies additional evaluation dimensions for later stages, including:

* calibration
* selective risk
* router regret
* coverage
* decision cost
* distribution shift
* uncertainty
* abstention

These are **not presented here as completed components**.

---

# Important Design Principle

AdaptiveML is not simply an ensemble that executes every model and votes.

The intended inference behavior is:

```text
              ┌── Logistic Regression
              │
              ├── Random Forest
Input → Router├── XGBoost
              ├── 1D CNN
              │
              └── LSTM
                    │
                    ▼
              Selected Model
                    │
                    ▼
                Prediction
```

The purpose of routing is to select a candidate rather than automatically executing the entire model pool for every input.

---

# Current Project Status

The current project has reached the stage where the main model pool, per-sample analysis, cost-aware routing, and inference pipeline have been implemented.

### Implemented

* HHAR data processing
* 2.5-second sensor windows
* group-aware user splitting
* tabular feature extraction
* Logistic Regression
* Random Forest
* XGBoost
* 1D CNN
* LSTM
* per-sample prediction storage
* error/disagreement analysis
* oracle analysis
* empirical latency benchmarking
* cost normalization
* cost-aware routing target
* XGBoost router
* router artifacts
* adaptive inference pipeline

### Next

**Deployment is the next stage.**

The project plan moves the trained AdaptiveML inference system toward a deployable application rather than treating the current local inference pipeline as the final product.

The deployment roadmap discussed in the project includes:

```text
AdaptiveML Inference Pipeline
            │
            ▼
         FastAPI
            │
            ▼
          Docker
            │
            ▼
        Deployment
```

Later project stages can build on this with the additional uncertainty, selective-decision, monitoring, and evaluation components discussed in the original roadmap.

---

# Repository Structure

The current project structure is:

```text
adaptive_ml/
│
├── .gitignore
├── DATA_CARD.md
├── README.md
├── requirements.txt
│
├── data/
│   ├── classical_model_predictions.csv
│   ├── deep_model_predictions.csv
│   │
│   ├── processed/
│   │   └── hhar_windows.npz
│   │
│   └── raw/
│       ├── Phones_accelerometer.csv
│   
│
├── models/
│   ├── 1d_cnn.pt
│   ├── logistic_regression.joblib
│   ├── lstm.pt
│   ├── random_forest.joblib
│   ├── router_label_encoder.pkl
│   ├── router_xgb.pkl
│   └── XGBoost.joblib
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
└── src/
    ├── analyze_full_pool.py
    ├── deep_models.py
    ├── evaluate_system.py
    ├── latency_benchmark.py
    ├── train_classical.py
    ├── train_deep_models.py
    ├── train_router.py
    │
    ├── data/
    │   └── preprocessing.py
    │
    └── pipeline/
        └── inference.py
```

The local `.venv` and generated `__pycache__` directories are not shown in the project tree above.

---

# Main Scripts

### `train_classical.py`

Trains:

```text
Logistic Regression
Random Forest
XGBoost
```

and saves their predictions and model artifacts.

### `train_deep_models.py`

Trains:

```text
1D CNN
LSTM
```

and saves their predictions and PyTorch model artifacts.

### `analyze_full_pool.py`

Builds the error matrix and calculates:

* individual model accuracy
* oracle accuracy
* all-model-wrong samples
* all-model-right samples
* disagreement samples

### `latency_benchmark.py`

Measures inference latency and creates normalized model-cost values.

### `train_router.py`

Creates the cost-aware routing target and trains the XGBoost meta-learner.

### `evaluate_system.py`

Evaluates the static and adaptive inference configurations.

### `pipeline/inference.py`

Loads the router and candidate models and performs adaptive model selection for a new sensor window.

---

# Technology Stack

The current implementation uses:

* Python
* NumPy
* Pandas
* Polars
* scikit-learn
* XGBoost
* PyTorch
* Joblib

---

# Core Research Question

The central question of AdaptiveML is:

> **When different models have different strength   s across individual samples, can a learned routing layer exploit those differences while accounting for inference cost?**

The current experiments establish the model pool, quantify model disagreement, measure the theoretical routing opportunity, introduce empirical inference cost, and implement a learned routing layer.

The next stage is to take this inference system toward **deployment**.
