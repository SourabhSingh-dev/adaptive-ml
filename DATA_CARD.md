# Data Card: HHAR Dataset

## 1. Provenance
* **Source:** UCI Machine Learning Repository (Heterogeneity Human Activity Recognition)
* **Collection Method:** 3D Accelerometer and Gyroscope sensor readings collected during physical activities.
* **Timeframe:** Data collected continuously at varying sampling rates.

## 2. Dataset Structure
* **Total Observations:** 13,062,475 raw accelerometer rows (prior to windowing).
* **Target Variable:** 6 distinct activities (Walk, Sit, Stand, Bike, Stairs Up, Stairs Down).
* **Unannotated Data:** 1,783,200 rows labeled 'null' (to be filtered out).
* **Label Distribution:** Highly balanced, ranging from ~1.6M (minority) to ~2.2M (majority) per class.

## 3. Heterogeneity Factors
* **Device Diversity:** 4 phone models, resulting in 8 distinct physical devices (e.g., Nexus 4, Samsung S3).
* **User Diversity:** 9 unique independent users.
* **Sensing Differences:** Hardware variance natively introduces distribution shifts in the continuous signals, acting as the primary catalyst for adaptive model routing.

## 4. Limitations & Missingness
* **Missing Values:** Zero missing values ($NaN$) in the continuous X, Y, Z sensor columns.and 