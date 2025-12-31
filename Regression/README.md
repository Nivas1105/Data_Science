# Regression Projects

A collection of machine learning regression projects focused on predicting house prices using various regression techniques and datasets.

## Overview

This directory contains multiple regression analysis projects that demonstrate different approaches to predicting real estate prices. The projects showcase data preprocessing, feature engineering, regression modeling, and evaluation techniques.

## Projects

### 1. Melbourne House Price Prediction

**File**: `house_price.ipynb`

A comprehensive Jupyter notebook analyzing Melbourne housing market data to predict property prices.

**Dataset**: 
- `MELBOURNE_HOUSE_PRICES_LESS.csv` (7.3 MB)
- Contains property features like location, rooms, property type, land size, etc.

**Key Features**:
- Exploratory data analysis (EDA) with visualizations
- Data cleaning and preprocessing
- Feature engineering and selection
- Multiple regression model comparison
- Model evaluation and validation

**Technologies**:
- pandas, numpy for data manipulation
- scikit-learn for machine learning models
- matplotlib, seaborn for visualizations

---

### 2. Kaggle Lasso Regression Challenge

**Directory**: `Kaggle_Challenge_Lasso_Regression/`

A focused project using Lasso Regression (L1 regularization) for house price prediction, following Kaggle competition format.

**Files**:
- `kaggle_challenge.py` - Main Python script with Lasso implementation
- `train.csv` - Training dataset
- `test.csv` - Test dataset

**Key Features**:
- Lasso Regression for feature selection and regularization
- Handles high-dimensional feature spaces
- Prevents overfitting through L1 penalty
- Optimized for Kaggle competition submission

**Model Characteristics**:
- Feature selection through coefficient shrinkage
- Automatic elimination of less important features
- Cross-validation for hyperparameter tuning
- Regularization parameter (alpha) optimization

---

## Regression Techniques Used

### Linear Regression
- Base model for comparison
- Simple interpretation of feature importance
- Establishes performance baseline

### Lasso Regression (L1 Regularization)
- Feature selection capability
- Reduces model complexity
- Prevents overfitting on high-dimensional data
- Some coefficients shrink to exactly zero

### Feature Engineering
- Handling missing values
- Categorical variable encoding
- Feature scaling and normalization
- Creating derived features
- Outlier detection and treatment

## Common Technologies

- **Python 3.x**
- **pandas & numpy** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **matplotlib & seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development

## How to Run

### Melbourne House Price Project

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook house_price.ipynb
   ```

2. **Run All Cells**: Execute sequentially to perform complete analysis

### Kaggle Lasso Challenge

1. **Navigate to Project Directory**:
   ```bash
   cd Kaggle_Challenge_Lasso_Regression
   ```

2. **Run Python Script**:
   ```bash
   python kaggle_challenge.py
   ```

## Evaluation Metrics

- **Mean Absolute Error (MAE)**: Average absolute difference between predictions and actual values
- **Mean Squared Error (MSE)**: Average squared difference (penalizes larger errors more)
- **Root Mean Squared Error (RMSE)**: Square root of MSE, in same units as target
- **R² Score**: Proportion of variance explained by the model (0 to 1, higher is better)

## Key Insights

1. **Feature Importance**: Location and property size are typically strong predictors
2. **Regularization Benefits**: Lasso helps prevent overfitting on complex datasets
3. **Data Quality**: Proper handling of missing values significantly impacts performance
4. **Model Selection**: Different datasets may benefit from different regression techniques

## Learning Outcomes

- Understanding regression fundamentals
- Handling real-world messy data
- Feature engineering techniques
- Model evaluation and selection
- Regularization for preventing overfitting
- Kaggle competition workflow

## Future Work

- Ensemble methods (Random Forest, Gradient Boosting)
- Ridge Regression (L2 regularization) comparison
- Elastic Net (L1 + L2 combined)
- Neural network regression models
- Time-series analysis for price trends
- Geospatial feature engineering

## License

Educational projects for data science portfolio.
