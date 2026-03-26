
import os

import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Check if required files exist
required_files = ['X_train.csv', 'y_train.csv', 'X_test.csv']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f"Missing files: {missing_files}")
else:
    print("All required files found. Loading data...")
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test.csv')
    print("Data loaded successfully.")