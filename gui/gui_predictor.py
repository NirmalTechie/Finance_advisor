import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

# Load model
model = joblib.load('models/expense_model.pkl')

# Prediction function
def predict():
    try:
        age = int(entry_age.get())
        income = float(entry_income.get())
        expenses = float(entry_expenses.get())
        savings = float(entry_savings.get())

        input_df = pd.DataFrame([[age, income, expenses, savings]],
                                columns=['Age', 'Income', 'Expenses', 'Savings'])
        predicted = model.predict(input_df)[0]
        result_label.config(text=f"Predicted Next Month's Expenses: ₹{predicted:.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI layout
root = tk.Tk()
root.title("💰 Expense Predictor")

tk.Label(root, text="Age").grid(row=0, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

tk.Label(root, text="Monthly Income").grid(row=1, column=0)
entry_income = tk.Entry(root)
entry_income.grid(row=1, column=1)

tk.Label(root, text="Current Expenses").grid(row=2, column=0)
entry_expenses = tk.Entry(root)
entry_expenses.grid(row=2, column=1)

tk.Label(root, text="Current Savings").grid(row=3, column=0)
entry_savings = tk.Entry(root)
entry_savings.grid(row=3, column=1)

tk.Button(root, text="Predict", command=predict).grid(row=4, column=0, columnspan=2)
result_label = tk.Label(root, text="")
result_label.grid(row=5, column=0, columnspan=2)

root.mainloop()
