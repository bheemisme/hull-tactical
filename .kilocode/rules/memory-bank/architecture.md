# Architecture

## System Architecture

The Hull Tactical Market Prediction system follows a modular, pipeline-based architecture designed for financial time series prediction with risk management constraints.

### Core Architecture Pattern

```
Data Layer → Processing Layer → Modeling Layer → Evaluation Layer → Deployment Layer
```

## Source Code Structure

### Root Level Files

- **`main.py`** - Main entry point (currently empty, planned as orchestration script)
- **`metric.py`** - Custom evaluation metric implementation with volatility-adjusted Sharpe ratio
- **`index.ipynb`** - Initial data exploration and analysis notebook
- **`agents.md`** - Project requirements and workflow documentation
- **`data_description.md`** - Comprehensive data description with target and feature definitions
- **`README.md`** - Project overview and Kaggle competition link
- **`schema.txt`** - Polars schema definition for data validation

### Data Layer

- **`data/`** - Contains training and test datasets
  - `train.csv` - Training dataset with 100+ features and target variables
  - `test.csv` - Test dataset for final predictions

### Memory Bank (`.kilocode/rules/memory-bank/`)

- **`brief.md`** - Project overview and core problem definition
- **`product.md`** - Product vision and user experience goals
- **`context.md`** - Current work focus and next steps
- **`architecture.md`** - System architecture documentation (this file)
- **`tech.md`** - Technology stack and development setup

## Key Technical Decisions

### 1. Data Processing Framework

**Decision**: Primary use of Polars for data manipulation with Pandas interoperability

- **Rationale**: Polars provides superior performance for large financial datasets
- **Implementation**: Data loading, schema validation, and feature engineering in Polars
- **Interoperability**: Pandas used for specific operations requiring pandas-specific libraries

### 2. Feature Engineering Strategy

**Decision**: Category-based feature processing pipeline

- **Categories**:
  - Technical/Market Dynamics (M): M1-M18
  - Macroeconomic (E): E1-E20
  - Interest Rate (I): I1-I9
  - Price/Valuation (P): P1-P13
  - Volatility (V): V1-V13
  - Sentiment (S): S1-S12
  - Momentum (MOM): M1-M18
  - Dummy/Binary (D): D1-D9
- **Approach**: Separate processing logic for each feature category
- **Validation**: Schema compliance maintained throughout transformations

### 3. Model Architecture

**Decision**: Multi-algorithm ensemble approach

- **Baseline Models**: Linear Regression, Random Forest (Scikit-learn)
- **Advanced Models**: XGBoost, LightGBM for gradient boosting
- **Ensemble Strategy**: Model stacking and blending for optimal performance

### 4. Evaluation Framework

**Decision**: Custom volatility-adjusted Sharpe ratio metric

- **Core Metric**: Sharpe ratio with volatility and return penalties
- **Position Constraints**: 0-2 range validation for realistic position sizing
- **Risk Management**: Excess volatility penalties above 1.2x market volatility

## Component Relationships

### Data Flow Architecture

```
Raw Train Data (CSV) → Polars Loading → Schema Validation → Feature Engineering →
Categorical Encoding → Feature Scaling → Train/Validation Split →
Model Training → Hyperparameter Tuning → Cross-Validation →
Performance Evaluation → Model Selection → Test Prediction (with Lagged Targets)
```

### Module Dependencies

- **Data Processing** → Independent, can operate standalone
- **Feature Engineering** → Depends on processed data schema
- **Model Training** → Depends on engineered features and evaluation metric
- **Evaluation** → Integrated across all components for consistent scoring

## Critical Implementation Paths

### 1. Data Pipeline Path

```
CSV Loading → Type Casting → Null Validation → Feature Categorization →
EDA Visualization → Statistical Analysis → Preprocessing → Feature Selection
```

### 2. Modeling Pipeline Path

```
Feature Matrix → Train/Val Split → Baseline Models → Advanced Models →
Hyperparameter Optimization → Ensemble Methods → Performance Comparison →
Final Model Selection
```

### 3. Evaluation Pipeline Path

```
Predictions → Strategy Returns Calculation → Sharpe Ratio Computation →
Volatility Analysis → Penalty Application → Adjusted Score → Model Ranking
```

## Design Patterns in Use

### 1. Pipeline Pattern

- **Purpose**: Modular, reusable processing steps
- **Implementation**: Each pipeline stage is independent and testable
- **Benefits**: Easy to modify, extend, and debug individual components

### 2. Factory Pattern

- **Purpose**: Model instantiation and configuration management
- **Implementation**: Centralized model creation with hyperparameter injection
- **Benefits**: Consistent model initialization and easy algorithm swapping

### 3. Strategy Pattern

- **Purpose**: Multiple evaluation metrics and model selection strategies
- **Implementation**: Pluggable metric calculations and model comparison methods
- **Benefits**: Easy to add new metrics or change selection criteria

## Component Interfaces

### DataProcessor Interface

- `load_data()` - Load and validate CSV data
- `preprocess_features()` - Apply category-specific transformations
- `generate_features()` - Create derived features and interactions

### ModelTrainer Interface

- `train_model()` - Train individual algorithms with cross-validation
- `tune_hyperparameters()` - Optimize model configurations
- `evaluate_performance()` - Calculate custom metric scores

### Evaluator Interface

- `calculate_sharpe()` - Compute volatility-adjusted Sharpe ratio
- `apply_penalties()` - Apply volatility and return penalties
- `compare_models()` - Rank and select best performing models

## Scalability Considerations

### Data Scale

- **Current**: 15+ years of daily data with 100+ features
- **Future**: Designed to handle additional data sources and extended time periods
- **Optimization**: Polars-based processing for memory efficiency

### Model Scale

- **Current**: Multiple algorithms with hyperparameter tuning
- **Future**: Ensemble methods and deep learning approaches
- **Optimization**: Parallel processing and model serialization

### Computational Scale

- **Current**: Single-machine processing with efficient libraries
- **Future**: Distributed computing for large-scale hyperparameter optimization
- **Optimization**: Vectorized operations and memory management
