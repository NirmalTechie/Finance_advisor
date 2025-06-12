from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/expense_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    data = {}
    if request.method == "POST":
        try:
            age = int(request.form["age"])
            income = float(request.form["income"])
            expenses = float(request.form["expenses"])
            savings = float(request.form["savings"])

            features = np.array([[age, income, expenses, savings]])
            prediction = round(model.predict(features)[0], 2)

            data = {
                "income": income,
                "expenses": expenses,
                "savings": savings,
                "prediction": prediction
            }

        except Exception as e:
            print(f"Error: {e}")

    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)
