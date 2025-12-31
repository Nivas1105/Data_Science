# Titanic Survival Prediction

A machine learning classification project that predicts passenger survival on the Titanic using Logistic Regression.

## Overview

This project analyzes the famous Titanic dataset to predict whether a passenger survived or not based on various features like age, gender, passenger class, and family relationships. The model uses feature engineering and logistic regression to achieve accurate predictions.

## Dataset

- **train.csv**: Training dataset with passenger information and survival outcomes
- **test.csv**: Test dataset for making predictions

### Features
- `PassengerId`: Unique identifier for each passenger
- `Survived`: Survival status (0 = No, 1 = Yes) - Target variable
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`: Passenger name
- `Sex`: Gender
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Passenger fare
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Methodology

### Data Preprocessing
1. **Missing Value Handling**:
   - Age: Filled with median age
   - Embarked: Filled with mode (most common port)
   - Dropped: Cabin, Ticket, and Name (high missing values or non-predictive)

2. **Feature Engineering**:
   - `TravelAlone`: Binary feature indicating if passenger traveled alone (SibSp + Parch = 0)
   - `IsMinor`: Binary feature indicating if passenger is 16 years old or younger

3. **Encoding**:
   - One-hot encoding for categorical variables (Sex, Embarked, Pclass)
   - Used `drop_first=True` to avoid multicollinearity

### Model
- **Algorithm**: Logistic Regression
- **Train/Test Split**: 80/20 (random_state=42)
- **Hyperparameters**: max_iter=1000
- **Feature Selection**: Recursive Feature Elimination (RFE)

### Evaluation Metrics
- Accuracy Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report
- ROC Curve visualization

## Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning algorithms and metrics
  - `seaborn` - Statistical data visualization
  - `matplotlib` - Plotting and visualization

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib
   ```

2. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook titanic.ipynb
   ```

3. **Run All Cells**: Execute cells sequentially to:
   - Load and explore the data
   - Preprocess and engineer features
   - Train the logistic regression model
   - Evaluate model performance
   - Visualize results

## Key Insights

- Gender is a strong predictor of survival (women had higher survival rates)
- Passenger class correlates with survival (1st class had better survival rates)
- Age and traveling alone status also influence survival probability
- Feature engineering improves model interpretability and performance

## Results

The model provides detailed classification metrics including accuracy, precision, recall, and F1-score, along with visualizations of the ROC curve to assess model performance across different classification thresholds.
