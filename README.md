
Project Overview
This project presents a comprehensive machine learning pipeline designed to predict customer churn using a large-scale synthetic dataset. By exploring the standard ML workflow and delving into the intricacies of modeling on artificially generated data, this project highlights the importance of data quality in driving model performance.
Key Contributions

    Developed a robust, modular pipeline incorporating state-of-the-art models (XGBoost, Logistic Regression, Random Forest) and thorough evaluation metrics (ROC-AUC, precision-recall)
    Conducted in-depth analysis of model performance, revealing limitations imposed by the synthetic dataset's lack of predictive power
    Demonstrated the significance of data signal integrity checks in machine learning pipelines

Findings and Insights
My experiments yielded several key takeaways:

    Despite utilizing advanced models, the pipeline failed to achieve satisfactory performance (AUC > 0.55) due to the dataset's artificial nature
    Exploratory data analysis revealed near-identical distributions and boxplots across churn classes, indicating a lack of meaningful separation
    The minor variation in one feature (~2-3%) was the only discernible signal

Technical Stack

    Language: Python
    ML Models: XGBoost, Logistic Regression, Random Forest
    Libraries: pandas, numpy, scikit-learn, xgboost
    Visualization: seaborn, matplotlib
    API Layer: FastAPI
    Containerization: Docker (planned)

Pipeline Architecture

    Data Loading: Efficient data ingestion and processing
    Exploratory Data Analysis: In-depth analysis of data distributions and relationships
    Feature Engineering: Scaling, cleaning, and transforming features for modeling
    Model Training: Cross-validation and hyperparameter tuning
    Model Evaluation: Comprehensive evaluation using ROC-AUC, precision-recall, and feature importance
    Failure Analysis: Investigating model limitations and data quality issues
    FastAPI Service: Deployable API for model serving

Evaluation Results
Model	AUC Score	Notes
Logistic Regression	~0.50	Baseline performance
Random Forest	~0.49	Marginal improvement
XGBoost	~0.51	Advanced model, no significant gain
Conclusion
This project underscores the critical importance of data quality in machine learning. By showcasing the limitations of advanced models on low-quality data, we demonstrate that ML pipelines alone cannot compensate for poor data signal integrity. As the adage goes: "In machine learning, data quality beats model complexity."
