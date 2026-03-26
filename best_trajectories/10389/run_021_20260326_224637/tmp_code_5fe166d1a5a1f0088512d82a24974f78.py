
import pandas as pd

import numpy as np

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print("=== TRAIN DATASET ===")
print(train.info())
print("\nMissing values in train:")
print(train.isnull().sum())
print("\nBasic statistics:")
print(train.describe())

print("\n=== TEST DATASET ===")
print(test.info())
print("\nMissing values in test:")
print(test.isnull().sum())
print("\nBasic statistics:")
print(test.describe())

print("\n=== SAMPLE SUBMISSION ===")
print(sample_submission.info())
print("\nFirst few rows:")
print(sample_submission.head())