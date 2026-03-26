
import pandas as pd

import numpy as np

# Reload datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Select only numerical columns for correlation
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns
print("\n=== CORRELATION MATRIX (NUMERICAL FEATURES) ===")
print(train[numerical_cols].corr())

# Check target distribution
print("\n=== TARGET VARIABLE STATISTICS ===")
print(train['target'].describe())

# Check for outliers in target
Q1 = train['target'].quantile(0.25)
Q3 = train['target'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("\n=== TARGET OUTLIERS ===")
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
print(f"Outliers below lower bound: {sum(train['target'] < lower_bound)}")
print(f"Outliers above upper bound: {sum(train['target'] > upper_bound)}")

# Check categorical feature distributions
categorical_features = train.select_dtypes(include=['object']).columns
print("\n=== CATEGORICAL FEATURE COUNTS ===")
for col in categorical_features:
    print(f"\n{col}:")
    print(train[col].value_counts().head(5))