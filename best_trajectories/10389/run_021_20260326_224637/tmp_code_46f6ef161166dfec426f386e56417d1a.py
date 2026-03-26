
import pandas as pd
from lightgbm import LGBMRegressor

# Reload data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Train LightGBM model with categorical features
model = LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel(), categorical_feature=categorical_cols)

# Make predictions on test set
predictions = model.predict(X_test)

# Create submission dataframe
submission = pd.DataFrame({
    'index': X_test.index,
    'prediction': predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission created successfully.")