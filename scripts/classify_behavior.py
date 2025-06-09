# scripts/classify_behavior.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

# Load dataset
data_path = r"C:\Users\nirma\OneDrive\Desktop\Personal-Finance-Advisor\data\personal_finance_data.csv"

df = pd.read_csv(data_path)

# Features and label
X = df[["Age", "Income", "Expenses", "Savings"]]
y = df["Category"]

# Encode labels manually
label_map = {"Saving": 0, "Balanced": 1, "Overspending": 2}
y_encoded = y.map(label_map)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

# Predict new input
print("\n🔍 Enter new person’s financial info:")
try:
    age = int(input("Age: "))
    income = float(input("Monthly Income: "))
    expenses = float(input("Monthly Expenses: "))
    savings = float(input("Current Savings: "))

    input_data = [[age, income, expenses, savings]]
    prediction = model.predict(input_data)[0]
    reverse_label_map = {v: k for k, v in label_map.items()}

    print(f"\n🧠 Predicted Financial Behavior: **{reverse_label_map[prediction]}**")
except ValueError:
    print("⚠️ Invalid input. Please enter numeric values only.")
