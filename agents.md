# Project Agents Context

## Project Overview

This project involves building a machine learning model from end-to-end, including:

1. **Data Collection** – Gathering data according to a defined schema.
2. **Data Preprocessing** – Cleaning, transforming, and normalizing data for model readiness.
3. **Data Analysis** – Descriptive analysis and identifying and selecting the most relevant features.
4. **Model Training** – Training models using suitable machine learning algorithms.
5. **Model Evaluation** – Evaluating the performance of models on validation data.
6. **Model Testing** – Testing the final model on a held-out test set to assess generalization.

The entire project is implemented in **Python**.

---

## Data

- The data has a **definite schema**.
- All transformations and analysis must respect the schema.
- Feature types should be correctly inferred (numerical, categorical, etc.).
- Handling missing values, outliers, and feature scaling is required.

### Data Directories

- Train: data/train.csv
- Test: data/test.csv

### Data Description

Contains all information about the data schema

- file path: data_description.md

---

## Libraries Used

### Data Preprocessing & Analysis

- **Polars** – primary library for fast data manipulation.
- **Pandas** – used where necessary, particularly for interoperability.
- **NumPy** – for numerical operations.
- **SciPy** – for statistical tests and advanced computations.
- **Matplotlib** – for data visualization.
- **Seaborn** – for statistical visualization.

### Machine Learning Model Training

- **Scikit-learn** – for traditional ML models, preprocessing, and evaluation.
- **XGBoost** – for gradient boosting tree-based models.
- **LightGBM** – for high-performance gradient boosting models.

---

## Project Workflow

1. **Data Collection**
   - Load data from CSV, JSON, or database sources.
   - Verify schema compliance.

2. **Data Preprocessing**
   - Handle missing values and duplicates.
   - Encode categorical variables.
   - Scale or normalize features if required.
   - Perform exploratory data analysis (EDA) using visualization libraries.

3. **Data Analysis**
    - Performing Descriptive analysis on the features of data set
    - Identify relevant features using correlation analysis, feature importance, or statistical tests.
    - Drop irrelevant or redundant features.

4. **Model Training**
   - Split data into training and validation sets.
   - Train multiple models using Scikit-learn, XGBoost, and LightGBM.
   - Tune hyperparameters using grid search, random search, or other optimization techniques.

5. **Model Evaluation**
   - Evaluate models using suitable metrics (accuracy, F1-score, RMSE, etc.).
   - Compare models and select the best-performing one.

6. **Model Testing**
   - Test the final model on a held-out test set.
   - Report performance metrics and interpret results.

---

## Coding Conventions

- Follow Python best practices (PEP8 compliance).
- Comment code clearly and concisely.
- Use meaningful variable and function names.
- Modularize code into functions or classes when necessary.

---

## Instructions for AI Coding Agents

- Always ensure transformations respect the original schema.
- When performing preprocessing, document the steps and reasoning.
- Visualizations should be clear, labeled, and insightful.
- Provide reasoning behind feature selection and model choice.
- Include hyperparameter tuning for all models.
- Output performance metrics after evaluation.
- Ensure reproducibility: set random seeds where applicable.

---

## Deliverables

- Preprocessed and clean dataset.
- Feature-selected dataset.
- Trained models (Scikit-learn, XGBoost, LightGBM).
- Evaluation reports and metrics.
- Test predictions and performance summary.
- All code documented and modularized in Python scripts or notebooks.
