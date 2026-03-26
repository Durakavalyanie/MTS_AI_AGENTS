
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Reload data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')

# Check data types
print("X_train dtypes:\n", X_train.dtypes)
print("\nX_test dtypes:\n", X_test.dtypes)

# Handle categorical variables
categorical_cols = X_train.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

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