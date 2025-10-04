# Tech

## Technologies Used

### Core Technologies

- **Python 3.x** - Primary programming language for all ML pipeline components
- **Polars** - High-performance DataFrame library for data manipulation and processing
- **Pandas** - Data manipulation and analysis library for interoperability requirements
- **NumPy** - Fundamental package for numerical computing and array operations
- **SciPy** - Scientific computing library for statistical tests and advanced computations

### Machine Learning Libraries

- **Scikit-learn** - Comprehensive ML library for traditional algorithms and preprocessing
  - Linear Regression, Random Forest, and other baseline models
  - Data preprocessing utilities (scaling, encoding, validation)
  - Model selection and evaluation tools
- **XGBoost** - Gradient boosting framework for high-performance tree-based models
- **LightGBM** - Microsoft gradient boosting framework optimized for speed and memory

### Visualization Libraries

- **Matplotlib** - Comprehensive plotting library for data visualization and analysis
- **Seaborn** - Statistical data visualization library built on matplotlib

### Development Environment

- **Jupyter Notebook** - Interactive computing environment for data exploration and analysis
- **VS Code** - Primary code editor with Python extensions and debugging capabilities

## Development Setup

### Environment Configuration

- **Operating System**: Windows 11 with PowerShell 7 as default shell
- **Python Version**: 3.8+ (compatible with all specified libraries)
- **Virtual Environment**: Recommended to use venv or conda for dependency isolation

### Required Dependencies Installation

```bash
# Core data processing
pip install polars pandas numpy scipy

# Machine learning
pip install scikit-learn xgboost lightgbm

# Visualization
pip install matplotlib seaborn

# Development tools
pip install jupyter notebook
```

### Project Structure Setup

```
hull_tactical/
├── data/                    # Dataset storage
│   ├── train.csv           # Training data with features and targets
│   └── test.csv            # Test data for final predictions
├── .kilocode/rules/memory-bank/  # Project documentation
├── main.py                 # Main orchestration script (planned)
├── metric.py               # Custom evaluation metric
├── index.ipynb             # Data exploration notebook
├── data_description.md     # Comprehensive data description
├── agents.md               # Project requirements and workflow
├── README.md               # Project overview and Kaggle link
└── schema.txt              # Data schema definition
```

## Technical Constraints

### Data Constraints

- **Schema Compliance**: All data transformations must respect the original Polars schema
- **Memory Efficiency**: Large dataset (15+ years daily data) requires efficient processing
- **Data Types**: Mixed data types (String, Int64, Float64) need proper handling
- **No Missing Values**: Dataset verified to have no null values

### Model Constraints

- **Position Limits**: Model predictions must be constrained to [0, 2] range
- **Evaluation Metric**: Custom volatility-adjusted Sharpe ratio with penalties
- **Reproducibility**: All models must use proper random seeds for consistent results
- **Cross-Validation**: Time series aware validation strategy required

### Performance Constraints

- **Training Time**: Models should train within reasonable timeframes
- **Memory Usage**: Efficient memory management for large feature sets (100+ features)
- **Scalability**: Framework should support future dataset expansion

## Dependencies

### Core Dependencies

- **polars**: Fast DataFrame library for data manipulation
- **pandas**: Data analysis library for specific operations
- **numpy**: Numerical computing package
- **scipy**: Scientific computing library

### ML Dependencies

- **scikit-learn**: Machine learning algorithms and utilities
- **xgboost**: Gradient boosting framework
- **lightgbm**: High-performance gradient boosting

### Visualization Dependencies

- **matplotlib**: Plotting and visualization library
- **seaborn**: Statistical data visualization

### Development Dependencies

- **jupyter**: Interactive computing environment
- **notebook**: Web-based interactive computing platform

## Tool Usage Patterns

### Data Processing Workflow

1. **Polars Primary**: Use Polars for all initial data loading and schema validation
2. **Pandas Interop**: Convert to Pandas only when required for specific library compatibility
3. **Memory Management**: Process data in chunks for large datasets when necessary

### Model Development Workflow

1. **Baseline First**: Start with simple Scikit-learn models for benchmarking
2. **Advanced Models**: Progress to XGBoost and LightGBM for performance improvements
3. **Ensemble Methods**: Combine multiple models for optimal results

### Feature Engineering Workflow

1. **Category-Specific**: Process each feature category (M, E, I, P, V, S) separately
2. **Schema Preservation**: Maintain original data types and constraints
3. **Validation**: Ensure all transformations preserve data integrity

## Development Best Practices

### Code Organization

- **Modular Design**: Separate concerns into distinct modules (data, models, evaluation)
- **Function Documentation**: Clear docstrings for all functions and classes
- **PEP8 Compliance**: Follow Python style guidelines for readability
- **Error Handling**: Proper exception handling for robust pipeline execution

### Version Control

- **Git Integration**: Project uses Git for version control and collaboration
- **Commit Strategy**: Regular commits with descriptive messages
- **Branch Management**: Feature branches for development, main for stable code

### Testing Strategy

- **Unit Tests**: Individual component testing for reliability
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Model training and inference time monitoring

## Future Technical Considerations

### Scalability Enhancements

- **Distributed Computing**: Framework ready for Dask or Spark integration
- **GPU Acceleration**: LightGBM and XGBoost support for GPU training
- **Model Serialization**: Pickle/joblib for model persistence and deployment

### Advanced Analytics

- **Feature Store**: Centralized feature management for reuse
- **Experiment Tracking**: MLflow or Weights & Biases integration potential
- **Model Interpretability**: SHAP values for feature importance analysis

### Production Deployment

- **API Development**: FastAPI or Flask for model serving
- **Containerization**: Docker for consistent deployment environments
- **Monitoring**: Performance and drift monitoring in production
