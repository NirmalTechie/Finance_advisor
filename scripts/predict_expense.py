# scripts/predict_expense.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load dataset
data_path = r"C:\Users\nirma\OneDrive\Desktop\Personal-Finance-Advisor\data\personal_finance_data.csv"

df = pd.read_csv(data_path)

# Features and target
X = df[["Age", "Income", "Expenses", "Savings"]]
y = df["NextMonthExpenses"]  # Make sure your CSV has this column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)

print("\n📊 Regression Model Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Predict new input
print("\n🔍 Enter new person’s financial info to predict next month expenses:")
try:
    age = int(input("Age: "))
    income = float(input("Monthly Income: "))
    expenses = float(input("Current Expenses: "))
    savings = float(input("Current Savings: "))

    input_data = [[age, income, expenses, savings]]
    predicted_expense = model.predict(input_data)[0]

    print(f"\n📈 Predicted Next Month's Expenses: ₹{predicted_expense:.2f}")
except ValueError:
    print("⚠️ Invalid input. Please enter numeric values only.")
