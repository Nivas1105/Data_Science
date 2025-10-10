import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

train = pd.read_csv('Kaggle_Challenge_Lasso_Regression/train.csv')
test = pd.read_csv('Kaggle_Challenge_Lasso_Regression/test.csv')  # Assuming test.csv exists for predictions

all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0, ignore_index=True)

quantitative = [f for f in all_data.columns if all_data.dtypes[f] != 'object' and f != 'Id']
qualitative = [f for f in all_data.columns if all_data.dtypes[f] == 'object']

for col in qualitative:
    if col in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 
               'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
               'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']:
        all_data[col] = all_data[col].fillna('None')
    else:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

for col in quantitative:
    if col in ['GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
               'TotalBsmtSF', 'LotFrontage']:
        all_data[col] = all_data[col].fillna(all_data[col].median())
    else:
        all_data[col] = all_data[col].fillna(0)

skewed_feats = all_data[quantitative].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_feats = skewed_feats[skewed_feats > 0.75].index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

train['SalePrice'] = np.log1p(train['SalePrice'])


def encode(frame, feature, target='SalePrice'):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, target]].groupby(feature).mean()[target]
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0] + 1)
    return ordering['ordering'].to_dict()

qual_encoded = []
for q in qualitative:
    ordering = encode(train, q)
    all_data[q + '_E'] = all_data[q].map(ordering).fillna(0)
    qual_encoded.append(q + '_E')

quadratic_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 
                      'GrLivArea', '1stFlrSF', '2ndFlrSF']
for feature in quadratic_features:
    all_data[feature + '2'] = all_data[feature] ** 2

boolean_features = {
    'HasBasement': lambda x: (x['TotalBsmtSF'] > 0).astype(int),
    'HasGarage': lambda x: (x['GarageArea'] > 0).astype(int),
    'Has2ndFloor': lambda x: (x['2ndFlrSF'] > 0).astype(int),
    'HasMasVnr': lambda x: (x['MasVnrArea'] > 0).astype(int),
    'HasWoodDeck': lambda x: (x['WoodDeckSF'] > 0).astype(int),
    'HasPorch': lambda x: (x['OpenPorchSF'] > 0).astype(int),
    'HasPool': lambda x: (x['PoolArea'] > 0).astype(int),
    'IsNew': lambda x: (x['YearBuilt'] > 2000).astype(int)
}
for name, func in boolean_features.items():
    all_data[name] = func(all_data)


all_data['OverallQual_KitchenQual'] = all_data['OverallQual'] * all_data['KitchenQual_E']
all_data['GarageArea_GarageCars'] = all_data['GarageArea'] * all_data['GarageCars']
all_data['1stFlrSF_TotalBsmtSF'] = all_data['1stFlrSF'] * all_data['TotalBsmtSF']

coords = all_data[['Neighborhood_E']].fillna(0)
kmeans = KMeans(n_clusters=5, random_state=42)
all_data['NeighborhoodCluster'] = kmeans.fit_predict(coords)


features = quantitative + qual_encoded + list(boolean_features.keys()) + \
           [f + '2' for f in quadratic_features] + \
           ['OverallQual_KitchenQual', 'GarageArea_GarageCars', '1stFlrSF_TotalBsmtSF', 'NeighborhoodCluster']


X_train = all_data.iloc[:train.shape[0]][features].fillna(0)
X_test = all_data.iloc[train.shape[0]:][features].fillna(0)
y_train = train['SalePrice']


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Ridge': RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f'{name} CV Log RMSE: {rmse_scores.mean():.4f} (± {rmse_scores.std():.4f})')

from sklearn.ensemble import StackingRegressor
stack = StackingRegressor(
    estimators=[(name, model) for name, model in models.items()],
    final_estimator=RidgeCV(cv=5)
)
stack.fit(X_train_scaled, y_train)
y_pred = stack.predict(X_train_scaled)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print(f'Stacked Model Train Log RMSE: {rmse:.4f}')

test_preds = np.expm1(stack.predict(X_test_scaled))

xgb_model = models['XGBoost']
xgb_model.fit(X_train_scaled, y_train)
feature_importance = pd.DataFrame({'feature': features, 'importance': xgb_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 10 Feature Importances (XGBoost)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=y_train - y_pred)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted Log(SalePrice)')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_train - y_pred, bins=50, kde=True)
plt.xlabel('Prediction Errors (Log Scale)')
plt.title('Distribution of Prediction Errors')
plt.show()

submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': test_preds})
submission.to_csv('submission.csv', index=False)