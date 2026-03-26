
import pandas as pd

# Check sample_submission.csv columns
sample_submission = pd.read_csv('sample_submission.csv')
print("Sample submission columns:", sample_submission.columns.tolist())
print("Sample submission shape:", sample_submission.shape)