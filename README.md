# 🧠 Customer Churn Prediction

This project focuses on predicting **customer churn** using various machine learning algorithms and performance optimization techniques.

---

## 📁 Project Overview

The goal is to build, tune, and compare multiple classification models to predict whether a customer will churn or not.
The dataset contains **505,000 samples** with a balanced target distribution.

---

## 🧾 Dataset

The dataset used in this project is publicly available on Kaggle:

📊 [Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

It contains **training** and **testing** CSV files (`customer_churn_dataset-training-master.csv` and `customer_churn_dataset-testing-master.csv`),
with a total of **505,000 records**. The target variable (`Churn`) is nearly balanced between classes 0 and 1.

---

## ⚙️ Models Used

Three optimized models were developed and compared:

| Model               | Accuracy | F1 Score | ROC-AUC |
| ------------------- | -------- | -------- | ------- |
| Logistic Regression | 0.815    | 0.826    | 0.887   |
| XGBoost             | 0.925    | 0.935    | 0.938   |
| LightGBM            | 0.925    | 0.935    | 0.938   |

---

## 📊 Confusion Matrices

**Logistic Regression**

```
           Pred 0  Pred 1
Actual 0   37972    6971
Actual 1   11713   44386
```

**XGBoost**

```
           Pred 0  Pred 1
Actual 0   38811    6132
Actual 1    1431   54668
```

**LightGBM**

```
           Pred 0  Pred 1
Actual 0   38812    6131
Actual 1    1435   54664
```

---

## 🔍 Insights

* **Logistic Regression** performed reasonably well but struggled with non-linear relationships in the data.
* **XGBoost** and **LightGBM** significantly outperformed Logistic Regression, with nearly identical results.
* Both tree-based methods achieved **high recall and precision**, indicating strong capability in handling large, complex datasets.
* **LightGBM** was slightly faster in training compared to XGBoost, making it more suitable for large-scale applications.

---

## 🧩 Techniques Applied

* Data balancing with **SMOTE**
* Feature selection based on correlation analysis
* Hyperparameter tuning with **GridSearchCV** and **RandomizedSearchCV**
* Evaluation metrics: Accuracy, F1 Score, ROC-AUC
* Visualization of confusion matrices and model comparison

---

## 🛠️ Tech Stack

* **Python 3.11**
* **scikit-learn**
* **xgboost**
* **lightgbm**
* **pandas**, **numpy**
* **matplotlib**, **seaborn**

---

## 📈 Future Work

* Apply **Stacking/Voting ensembles** for potential F1 improvement
* Experiment with **CatBoost** and deep learning models (MLP/ANN)
* Optimize training efficiency using GPU acceleration

---

## 👤 Author

**Sinan Ustun**
Machine Learning Engineer
🔗 [LinkedIn](https://www.linkedin.com/in/sinan-ustun/) | [GitHub](https://github.com/Sinan-Ustun)
