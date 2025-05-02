# churn-prediction-pipeline-synthetic

Customer Churn Prediction – Production Pipeline with Synthetic Data
Status: Completed
Dataset: Kaggle: Customer Churn in Large Datasets
Goal: Build a production-grade machine learning pipeline for customer churn prediction and investigate the challenges of modeling on synthetic data.

Project Summary
This project is an end-to-end machine learning pipeline built on a large-scale synthetic dataset that simulates customer churn behavior. It explores not only the standard ML workflow (EDA → Feature Engineering → Modeling → Evaluation), but also showcases:

Robust pipeline design

Model performance analysis

Data signal integrity checks

Why even the most advanced models fail when the dataset lacks true predictive power

Key Findings
Despite applying state-of-the-art models, the project surfaced a crucial insight:

The dataset appears to be artificially generated and lacks meaningful separation between churned and non-churned customers.

Boxplots, distributions, and hist plots show nearly identical behavior across churn classes.

Models such as XGBoost, Logistic Regression, and Random Forest failed to achieve AUC > 0.55.

Minor variation (~2–3%) in one feature was the only visible signal.

Stack Used

Component	Tech
Language	Python
ML Models	XGBoost, Logistic Regression, Random Forest
Libraries	pandas, numpy, scikit-learn, xgboost
Visualization	seaborn, matplotlib
API Layer	FastAPI 
Containerization	Docker (future goal)
Deployment Ready?	Yes – ML pipeline modular 


Pipeline Overview

1. Data Loading
2. Exploratory Data Analysis
3. Feature Engineering (scaling, cleaning)
4. Model Training (cross-validation)
5. Model Evaluation (ROC, AUC, precision-recall)
6. Failure Analysis (box plots, feature importance)
7. FastAPI service

   
Evaluation Results

Model	AUC Score	Notes
Logistic Regression	~0.50	Balanced, simple linear model
Random Forest	~0.49	Marginally better, but still underperforms
XGBoost	~0.51	Advanced model, but no improvement

Why the Model Fails (And Why That’s the Point)

Balanced target variable: Churn rate = 50%

No class imbalance – yet models perform poorly

No feature separates the churn class meaningfully

Demonstrates that ML pipelines alone can’t compensate for low-quality signal

"In machine learning, data quality beats model complexity."

