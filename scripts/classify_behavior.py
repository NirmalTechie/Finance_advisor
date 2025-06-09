import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Path to data
data_path = os.path.join('data', 'personal_behavior_data.csv')

# Load data
df = pd.read_csv(data_path)

# Features and target
X = df[["Age", "Income", "Expenses", "Savings"]]
y = df["Category"]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Save the label map
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
joblib.dump(le, 'models/behavior_label_encoder.pkl')

# Split data with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/behavior_classifier.pkl')

# Report
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
