import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
data_path = os.path.join('data', 'personal_finance_data.csv')
df = pd.read_csv(data_path)

# Define rules to create labels (if not present)
def classify_behavior(row):
    if row['Expenses'] > row['Income'] * 0.9:
        return 'Overspending'
    elif row['Savings'] > row['Income'] * 0.2:
        return 'Saving'
    else:
        return 'Balanced'

# Create label column
df['Behavior'] = df.apply(classify_behavior, axis=1)

# Features and label
X = df[['Age', 'Income', 'Expenses', 'Savings']]
y = df['Behavior']

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/behavior_classifier.pkl')

print("✅ Classifier trained and saved as models/behavior_classifier.pkl")
