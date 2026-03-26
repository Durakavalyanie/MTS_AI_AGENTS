
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Reload data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')

# Select only numeric columns
numeric_cols = X_train.select_dtypes(include=['number']).columns
X_train_numeric = X_train[numeric_cols]
X_test_numeric = X_test[numeric_cols]

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_numeric, y_train.values.ravel())

# Make predictions on test set
predictions = model.predict(X_test_numeric)

# Create submission dataframe
submission = pd.DataFrame({
    'index': X_test.index,
    'prediction': predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission created successfully.")