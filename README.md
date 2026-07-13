# Model Training on Two Datasets with Comparison
# BetaBytez AI/ML Track — Week 2 Task

Welcome to my Week 2 portfolio project! In this task, I took on the role of a Data Scientist to analyze, preprocess, train, and compare machine learning models across two entirely different real-world classification domains.
# Datasets Chosen
1. Bank Customer Churn Dataset:
    Goal: Predict whether a customer will leave the bank or stay.
    Target Variable: `churn` (Binary: 1 = Left, 0 = Stayed)
    Key Features: Credit score, age, balance, active membership, geography, gender.
    Challenge: Notable class imbalance (20% churned vs. 80% stayed).

2. Heart Disease Dataset:
    Goal: Predict the presence of heart disease in patients.
    Target Variable: `HeartDisease` (Binary: 1 = Disease, 0 = Healthy)
    Key Features: Age, resting blood pressure, cholesterol, maximum heart rate, physical indicators.
    Characteristics: Highly distinctive clinical correlations.

# Key Findings & Performance Comparison
I trained two distinct classifiers Logistic Regression and Random Forest on both datasets. Due to severe class imbalance in the Bank dataset, I also applied hyperparameter tuning via `GridSearchCV` to improve the weaker model.

# Model Performance Summary Table

| Dataset | Model | Accuracy | Precision | Recall | F1-Score |
| :---    | :---  | :---:    | :---:     | :---:  | :---:    |

| Bank Churn |     Logistic Regression   | 80.70% | 58.02% | 18.67% | 0.2825 |
| Bank Churn |     Random Forest         | 85.90% | 76.37% | 44.47% | 0.5621 |
| Bank Churn | Tuned Logistic Regression | 70.65% | 37.87% | 69.04% | 0.4891 |

| Heart Disease |        Logistic Regression         | 92.38%  | 91.73%  | 91.84%  | 0.9179 |
| Heart Disease | Random Forest (Perfect Classifier) | 100.00% | 100.00% | 100.00% | 1.0000 |

# Key Insights & Takeaways

The Harder Domain: The Bank Churn dataset was significantly harder to model. Because of the class imbalance, baseline models optimized for overall accuracy while completely missing actual churned customers (Recall of only `18.67%` on baseline Logistic Regression).

The Power of Hyperparameter Tuning: By running `GridSearchCV` with `class_weight='balanced'` on the weaker Bank Logistic Regression model, I successfully forced the model to penalize minority class errors. This massive engineering adjustment raised the Recall from 18.67% to 69.04% and improved the F1-Score from 0.2825 to 0.4891.

Generalization Champion: Random Forest emerged as the superior model across both datasets. It handled the non-linear boundaries of the medical dataset flawlessly (achieving `100%` metrics) and outclassed Logistic Regression on the financial dataset without requiring scaling adjustments.