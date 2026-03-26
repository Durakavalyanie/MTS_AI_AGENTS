
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Reload data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')

# One-hot encode all categorical variables
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Ensure test set has same columns as train set
common_columns = X_train_encoded.columns.intersection(X_test_encoded.columns)
X_test_encoded = X_test_encoded[common_columns]

# Fill missing columns in test set with 0
missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for col in missing_cols:
    X_test_encoded[col] = 0
X_test_encoded = X_test_encoded[X_train_encoded.columns]

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_encoded, y_train.values.ravel())

# Make predictions on test set
predictions = model.predict(X_test_encoded)

# Create submission dataframe
submission = pd.DataFrame({
    'index': X_test.index,
    'prediction': predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission created successfully.")