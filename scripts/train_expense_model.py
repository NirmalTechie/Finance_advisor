import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load data
data_path = os.path.join('data', 'personal_finance_data.csv')
df = pd.read_csv(data_path)

# Features and target
X = df[['Age', 'Income', 'Expenses', 'Savings']]
y = df['NextMonthExpenses']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, os.path.join('models', 'expense_model.pkl'))

print("✅ Model trained and saved as models/expense_model.pkl")
