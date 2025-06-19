## Employee Attrition Prediction (Logistic Regression) ##

This project predicts whether an employee is likely to leave the company using Logistic Regression. The model is optimized to maximize **recall for class 1 (attrition cases)**, making it valuable for HR departments aiming to proactively retain employees.

---

## Dataset

- **Source:** IBM HR Analytics Employee Attrition & Performance dataset
- **Link:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data
- **Target variable:** `Attrition` (0 = No, 1 = Yes)

---

## Challenges Faced
- The Dataset contains many features,proper feature engineering to be done.
- Many categorical features needed to be encoded properly, and handling them without causing dimensionality issues was challenging.
- Accuracy was misleading; better metrics were needed.
- Balancing precision and recall was challenging.
- imbalanced Attrition data, maks it challenging to maximize the recall.
- Preventing overfitting while maintaining performance was tricky.

---

## Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Model Overview

- **Model Used:** Logistic Regression  
- **Scaling:** `StandardScaler`  
- **Custom Thresholding:** Yes (threshold adjusted from 0.5 to prioritize recall)  
- **Evaluation:** Focused on recall for class 1 (attrition)

---

## Final Model Performance

precision    recall  f1-score   support

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| **0** | 0.95      | 0.77   | 0.85     | 255     |
| **1** | 0.33      | 0.74   | 0.46     | 39      |
|       |           |        |          |         |
| **Accuracy**     |         |          | **0.77** | 294     |
| **Macro Avg**    | 0.64    | 0.76     | 0.65     | 294     |
| **Weighted Avg** | 0.87    | 0.77     | 0.80     | 294     |

Focus: High recall for attrition (class 1) to minimize false negatives.

---

## How It Works

1. Load and preprocess HR attrition dataset
2. One-hot encode categorical features
3. Scale features using `StandardScaler`
4. Train Logistic Regression model
5. Adjust decision threshold using `predict_proba` for better recall
6. Evaluate with classification report and visualize feature importance

---

## Key Learnings

- Importance of scaling in logistic regression
- Trade-off between precision and recall
- Accuracy misleads in case of im-balanced data
- Custom thresholding improves minority class detection

---
## Files
- `employee_attrition_logistic.ipynb` – The main Jupyter notebook with data preprocessing, logistic regression model, threshold tuning, and evaluation.
- `feature_importance.png` – Image showing the logistic regression feature importances.
- `classification_report.png` – Screenshot of the final classification report.

