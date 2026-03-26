
import pandas as pd

import numpy as np
from sklearn.preprocessing import LabelEncoder

# Read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Check column names
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

# Separate target variable
y_train = train['target']
X_train = train.drop(columns=['target'])

# Handle missing values
# For last_dt: fill with mode (most frequent value)
last_dt_mode = X_train['last_dt'].mode()[0]
X_train['last_dt'] = X_train['last_dt'].fillna(last_dt_mode)
test['last_dt'] = test['last_dt'].fillna(last_dt_mode)

# For avg_reviews: fill with median
avg_reviews_median = X_train['avg_reviews'].median()
X_train['avg_reviews'] = X_train['avg_reviews'].fillna(avg_reviews_median)
test['avg_reviews'] = test['avg_reviews'].fillna(avg_reviews_median)

# Check for categorical columns
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

# Encode categorical variables
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Create additional features
# Extract year from last_dt
X_train['last_dt_year'] = pd.to_datetime(X_train['last_dt']).dt.year
test['last_dt_year'] = pd.to_datetime(test['last_dt']).dt.year

# Create interaction features
X_train['price_per_review'] = X_train['price'] / (X_train['avg_reviews'] + 1)
test['price_per_review'] = test['price'] / (test['avg_reviews'] + 1)

X_train['host_ratio'] = X_train['total_host'] / (X_train['minimum_nights'] + 1)
test['host_ratio'] = test['total_host'] / (test['minimum_nights'] + 1)

# Drop original last_dt column
X_train = X_train.drop(columns=['last_dt'])
test = test.drop(columns=['last_dt'])

# Save files
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
test.to_csv('X_test.csv', index=False)