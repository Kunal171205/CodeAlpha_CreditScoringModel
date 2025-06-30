# 📊 Credit Score Prediction using XGBoost

This project focuses on building a machine learning model to predict the **creditworthiness** of loan applicants using **XGBoost**. The notebook includes data preprocessing, encoding, resampling with SMOTE, model training, and various performance visualizations.

---

## 📁 Files

- `codealpha-creditscoremodel.ipynb` – Main Jupyter notebook containing the end-to-end pipeline for credit score classification.

---

## 🧾 Dataset Features

The dataset includes the following features:

| Feature     | Description                        |
|-------------|------------------------------------|
| `Cbal`      | Checking account balance           |
| `Cdur`      | Credit duration                    |
| `Chist`     | Credit history                     |
| `Cpur`      | Purpose of the loan                |
| `Camt`      | Loan amount                        |
| `Sbal`      | Savings balance                    |
| `Edur`      | Employment duration                |
| `InRate`    | Installment rate                   |
| `MSG`       | Marital status & gender            |
| `Oparties`  | Other parties involved             |
| `Rdur`      | Residence duration                 |
| `Prop`      | Property type                      |
| `Age`       | Applicant age                      |
| `inPlans`   | Existing plans                     |
| `Htype`     | Housing type                       |
| `NumCred`   | Number of credits                  |
| `JobType`   | Job type                           |
| `Ndepend`   | Number of dependents               |
| `telephone` | Has telephone                      |
| `foreign`   | Foreign worker status              |
| `creditScore` | 🔴 Target: Creditworthy or not |

---

## 🧠 Model Pipeline

1. **Preprocessing**
   - Label Encoding for binary categorical features
   - One-Hot Encoding for multi-class features
2. **Resampling**
   - Handled class imbalance using **SMOTE** 
3. **Model**
   - Trained with `XGBClassifier`
4. **Evaluation**
   - Confusion Matrix (raw and normalized)
   - Accuracy, Precision, Recall, F1-score
   - Feature Importance Plot

---

## 📈 Visualizations Included

- Confusion Matrix Heatmap
- Feature Importance Bar Plot
- Prediction Probability Distribution
- Class Balance Before/After Resampling

---

## 🔧 Requirements

Install all dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
