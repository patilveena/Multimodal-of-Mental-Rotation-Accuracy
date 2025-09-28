# Multimodal Prediction of Mental Rotation Accuracy  

<div align="center">
<h5><b>IITB EdTech Internship 2025</b> | DYPCET <br>Track 1 – Educational Data Analysis (EDA)</h5>  
</div>

---

**Group Name**: Data Divas  
**Group ID**: T1_G23  

---

## 🚀 Project Overview
Human cognition is complex, but measurable. This project explores how physiological and behavioral signals can be used to **predict performance in mental rotation tasks** a classic test of spatial reasoning.  

By combining signals from **EEG (brain activity)**, **eye-tracking**, **GSR (skin response)**, and **facial expressions**, we aim to build machine learning models that can forecast whether a participant will answer correctly **before or during the task itself**.  

### Why it matters
- Mental rotation tests capture spatial intelligence and cognitive load.  
- EEG, gaze behavior, and skin response often correlate with task difficulty and accuracy.  
- Multimodal models combine strengths of different sensors, boosting predictive power.  

---

## 🎯 Problem Statement
We frame the task as a **binary classification** problem:  

> **Predict if a participant’s response to a mental rotation question is Correct (1) or Incorrect (0).**

**Modalities used:**  
- **EEG**: Delta, Theta, Alpha, Beta, Gamma bands  
- **Eye-tracking**: Fixations, saccades, pupil dilation, gaze spread  
- **GSR**: Skin conductance & resistance  
- **Facial expressions (TIVA)**: Emotion scores, Action Units  

**Models explored:**  
- Baselines: Random Forest, Logistic Regression, XGBoost  
- Modality-wise models + Fusion via stacking/ensemble  

**Goal:** Build a high-performing, interpretable predictor while analyzing which signals matter most.  

---

## 📊 Dataset Description
The dataset is multimodal, synchronized on question timestamps.  

| File | Description |
|------|-------------|
| PSY.csv | Target variable (Correct/Incorrect) + timestamps |
| EEG.csv | EEG band-level features |
| EYE.csv & IVT.csv | Eye-tracking metrics (fixations, saccades, gaze spread) |
| GSR.csv | Skin conductance/resistance |
| TIVA.csv | Facial emotions + Action Units |

**Preprocessing highlights**  
- Time-align modalities per question  
- Statistical aggregation: mean, max, min, std  
- Missing values handled, imbalance corrected with **SMOTE**  

---

## 🛠 Methodology

### Step 1: Preprocessing
- Merge multimodal features per question  
- Standardize using `StandardScaler`  
- Encode target (Correct=1, Incorrect=0)  
- Balance with **SMOTE**  

### Step 2: Feature Engineering
- **EEG**: mean & variance across frequency bands  
- **GSR**: peaks, average, temporal changes  
- **Eye**: fixation duration, saccade amplitude, pupil dilation  
- **Facial**: mean AU activations, emotion intensity  

### Step 3: Modeling
- **Baselines**: Random Forest, XGBoost, Logistic Regression  
- **Hyperparameter tuning**: with **Optuna**, maximizing F1-score  
- **Fusion models**: Modality-specific models fused via stacking/ensembles  

### Step 4: Evaluation & Interpretability
- Metrics: Accuracy, F1-score, ROC-AUC, Confusion Matrix  
- Tools:  
  - **SHAP** → global feature importance  
  - **LIME** → local prediction explanations  

### Step 5: Model Selection
- Best model chosen via F1-score  
- Saved models for deployment:  
  ```
  xg_model.pkl
  scaler.pkl
  fusion_model.pt
  ```

---

## 🧩 Modeling Approaches

### 1️⃣ Baseline Models
- **Random Forest** → captures non-linear relations  
- **XGBoost** → gradient boosting with regularization  
- **Logistic Regression** → simple linear baseline  

Hyperparameter optimization with **Optuna**.  
Outcome: RF and XGBoost > Logistic Regression in performance.  

### 2️⃣ Multimodal Fusion
- Train separate models (EEG, Eye, GSR, Facial)  
- Fuse logits/embeddings with stacking ensemble  
- Interpretability: SHAP + LIME showed **EEG and Eye-tracking dominate predictions**, followed by GSR & Facial.  

---

## 🧪 Evaluation Metrics
- **Accuracy** → % correct predictions  
- **Precision / Recall** → balance false positives & negatives  
- **F1-score** → harmonic mean of precision & recall  
- **ROC-AUC** → discrimination ability  
- **Confusion Matrix** → class-wise performance  

All evaluations done with cross-validation + held-out test splits.  

---

## 📂 Repository Structure

```
project/
├── data/
│ ├── PSY_feature_engineered.csv
│ ├── EEG_feature_engineered.csv
│ ├── GSR_feature_engineered.csv
│ ├── EYE_feature_engineered.csv
│ ├── IVT_feature_engineered.csv
│ └── TIVA_feature_engineered.csv
├── notebooks/
│ ├── 01_preprocessing.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_modeling_baseline.ipynb
│ ├── 04_modeling_fusion.ipynb
│ └── 05_analysis.ipynb
├── models/
│ ├── scaler.pkl
│ ├── xgb_model.pkl
│ └── fusion_model.pt
└── README.md
```
