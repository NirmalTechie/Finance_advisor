import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load model
model = joblib.load('models/behavior_classifier.pkl')

def predict_behavior():
    try:
        age = int(age_entry.get())
        income = float(income_entry.get())
        expenses = float(expenses_entry.get())
        savings = float(savings_entry.get())

        input_data = np.array([[age, income, expenses, savings]])
        prediction = model.predict(input_data)[0]

        messagebox.showinfo("Prediction", f"🧠 Spending Behavior: {prediction}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values.")

# GUI setup
root = tk.Tk()
root.title("💡 Spending Behavior Classifier")

tk.Label(root, text="Age").grid(row=0, column=0)
tk.Label(root, text="Monthly Income").grid(row=1, column=0)
tk.Label(root, text="Current Expenses").grid(row=2, column=0)
tk.Label(root, text="Current Savings").grid(row=3, column=0)

age_entry = tk.Entry(root)
income_entry = tk.Entry(root)
expenses_entry = tk.Entry(root)
savings_entry = tk.Entry(root)

age_entry.grid(row=0, column=1)
income_entry.grid(row=1, column=1)
expenses_entry.grid(row=2, column=1)
savings_entry.grid(row=3, column=1)

tk.Button(root, text="Classify", command=predict_behavior).grid(row=4, column=0, columnspan=2)

root.mainloop()
