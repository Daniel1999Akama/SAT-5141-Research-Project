# 🔍 Explainable AI for Real-Time Clinical Decision Support in Predicting 30-Day Readmission for Diabetic Patients.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Research%20Prototype-yellow)
![Models](https://img.shields.io/badge/Models-LR%20%7C%20RF%20%7C%20MLP%20%7C%20XGBoost-red)
![SMOTE](https://img.shields.io/badge/SMOTE-Applied-blueviolet)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-red)

This project develops machine learning models to predict hospital readmissions using real-world clinical encounter data and explains model behavior using SHAP.  
A UI was **not** implemented due to moderate model performance, focusing instead on explainability and model diagnostics.

---

## 1. Project Overview

The goal is to predict hospital readmissions categorized as:

- **0** → No readmission  
- **1** → Readmitted within 30 days  
- **2** → Readmitted after 30 days  

The project focuses on:

- Patient-level train/test splits (no leakage)  
- Class balancing with SMOTE  
- Feature engineering for clinical interpretability  
- Machine learning experimentation with four models  
- SHAP-based global + local interpretability  

---

## 2. Dataset

**Source:** UCI Diabetes Readmission Dataset  
🔗 https://archive.ics.uci.edu/dataset/296/diabetic+data

Key properties:

- ~100,000 hospital encounters  
- Unique `patient_nbr` allows patient-level splitting  
- Contains diagnoses, medications, demographics, and lab measures  
- Three-class readmission target  

**Validation checks done:**

- Missingness patterns  
- Outlier detection  
- Duplicate encounter removal  
- Logical range checks  
- Diagnosis and medication consistency  

---

## 3. Workflow Summary

### 3.1 Data Cleaning & Initial Preprocessing
- Standardized column names  
- Handled missing values  
- Recoded categories  
- Removed invalid/outlier values  
- Filtered implausible encounter records  

---

### 3.2 🧪 Exploratory Data Analysis (EDA)

Performed **after initial preprocessing**.

**Univariate Analysis:**  
- Numerical distributions  
- Categorical counts  
- Demographic summaries  

**Bivariate Analysis:**  
- Correlations
![alt text](https://github.com/Daniel1999Akama/SAT-5141-Research-Project/blob/main/screenshots/correlation.png)

- Feature vs. target relationships

**Class imbalance identified (training set):**

| Class | Count |
|-------|-------|
| 0 | 8,655 |
| 1 | 27,274 |
| 2 | 41,794 |

---

### 3.3 🔧 Feature Engineering

Added interpretable clinical features:

- Age brackets  
- Comorbidity score  
- Total medications    
---

### 3.4 🔐 Patient-Level Train/Test Split

Used **GroupShuffleSplit** to ensure:

- Same patient **never** appears in both sets (train and test)  
- No leakage from repeated encounters  
- More realistic generalization performance  

---

### 3.5 🧬 Balancing With SMOTE

Applied **only on training data** to avoid leakage.

Balanced class counts:  
→ **All classes oversampled to 41,794**  

![alt text](https://github.com/Daniel1999Akama/SAT-5141-Research-Project/blob/main/screenshots/smote.png)
---

### 3.6 📏 Scaling

- **Scaled:** Logistic Regression, MLP  
- **Unscaled:** Random Forest, XGBoost  

---

### 3.7 ⚙️ Model Training

Models trained:

1. Logistic Regression  
2. Random Forest (GridSearchCV)  
3. MLP Classifier (GridSearchCV)  
4. XGBoost Classifier (GridSearchCV)

---

### 3.8 📊 Evaluation

Reported (Macro):

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC AUC  

![alt text](https://github.com/Daniel1999Akama/SAT-5141-Research-Project/blob/main/screenshots/metrics.png)

**Best model:**  
⭐ **XGBoost** — AUC ~0.66  
But overall performance remains moderate.

---

### 3.9 🧠 Explainable AI (SHAP)

**Global Explanations:**  
- Summary SHAP plots
![alt text](https://github.com/Daniel1999Akama/SAT-5141-Research-Project/blob/main/screenshots/shap.png) 
- Feature importance rankings
![alt text](https://github.com/Daniel1999Akama/SAT-5141-Research-Project/blob/main/screenshots/importance.png) 
- Feature dependence  

**Local Explanations:**  
- Sample-level waterfall plots  
![alt text](https://github.com/Daniel1999Akama/SAT-5141-Research-Project/blob/main/screenshots/waterfall.png) 

Strong emphasis on **interpretability over deployment**.

---

## 4. Results Summary

| Model | AUC Macro | Notes |
|--------|------------|-------|
| Logistic Regression | ~0.59 | Weak signal |
| Random Forest | ~0.65 | Better but unstable |
| MLP Classifier | ~0.62 | Overfits easily |
| **XGBoost** | **~0.66** | Best performer |

**Conclusion:** Predictions remain noisy and unreliable for clinical deployment.

---

## 5. Why No UI Was Built

Building a clinical UI requires:

- High recall for critical cases  
- Strong predictive power  
- Explainability and stability  
- Low risk of misclassification  

Since the models achieve only **moderate accuracy and AUC**, deploying a UI would create a false sense of reliability.  
Thus, we prioritized **explainability, diagnostics, and data understanding** instead.

---

## 6. Next Steps

### ⭐ Data Enhancements
- Add temporal features  
- Incorporate longitudinal patient histories  
- Extract richer comorbidity metrics  
- Use visit sequences  

### ⭐ Modeling Enhancements
- Try LSTM/GRU sequence models  
- Feature selection  
- Calibration (Platt scaling, isotonic regression)  

### ⭐ XAI Enhancements
- SHAP interaction values  
- Counterfactual explanations  
- Monitoring & drift detection  

### ⭐ Deployment (After Further Improvements)
- UI for real-time CDS  
- Workflow-integrated alerts  
- Model monitoring dashboards  

---

## 7. Installation and requirements
Requirements - ensure you have python and VS code installed with the following libraries.
- Python 3.10+, pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap, matplotlib, seaborn

Setup
- Take repo URL then in terminal:
> git clone "repo URL"
- Open Jupyter Notebook in VS code and run all cells.

## 8. Contributors
Daniel Nyamweya (Lead Researcher & Implementer)



