import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("16P.csv", encoding="ISO-8859-1")  # Ensure correct encoding

# Extract features & target
X = df.drop(columns=["Response Id", "Personality"], errors="ignore")
y = df["Personality"]

# Train model (Use best params)
best_params = {
    "n_estimators": 150,
    "max_depth": 17,
    "min_samples_split": 6,
    "min_samples_leaf": 2,
    "random_state": 42
}
rf = RandomForestClassifier(**best_params)
rf.fit(X, y)

# Select Top 25 Features
importances = rf.feature_importances_
top_n = 25  # Number of features to keep
selected_features = X.columns[np.argsort(importances)[::-1][:top_n]]

# Initialize Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", features=selected_features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure all required inputs are received
        user_input = {}
        for feature in selected_features:
            if feature not in request.form:
                return jsonify({"error": f"Missing input for {feature}"}), 400
            user_input[feature] = float(request.form[feature])

        # Convert to DataFrame
        input_data = pd.DataFrame([user_input])

        # Make prediction
        predicted_label = rf.predict(input_data)[0]

        return render_template("index.html", features=selected_features, result=predicted_label)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT variable
    app.run(host="0.0.0.0", port=port, debug=True)
