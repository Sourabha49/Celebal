import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

train = pd.read_csv(r"E:\Kama_Karma_&_Work\Celebal Ass\Assignment 5\train.csv")
test = pd.read_csv(r"E:\Kama_Karma_&_Work\Celebal Ass\Assignment 5\test.csv")

train_ID = train['Id']
test_ID = test['Id']
y = train['SalePrice']
train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

data = pd.concat([train, test], axis=0)

none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']
for col in none_cols:
    data[col] = data[col].fillna("None")

num_fill_median = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']
for col in num_fill_median:
    data[col] = data[col].fillna(data[col].median())

cat_cols = data.select_dtypes(include='object').columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

label_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
              'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
for col in label_cols:
    lbl = LabelEncoder()
    data[col] = lbl.fit_transform(data[col])

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['TotalBath'] = data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath'])
data['HasPool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
data['HasGarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

skewed_feats = data.select_dtypes(include=[np.number]).apply(lambda x: x.skew()).sort_values(ascending=False)
skewed = skewed_feats[abs(skewed_feats) > 0.75].index
for feat in skewed:
    data[feat] = np.log1p(data[feat])

data = pd.get_dummies(data)
X_train = data.iloc[:train.shape[0], :]
X_test = data.iloc[train.shape[0]:, :]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocessing complete.")
print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)
