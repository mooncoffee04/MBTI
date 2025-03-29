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
        # Get user inputs for only the selected 25 features
        user_input = {feature: float(request.form[feature]) for feature in selected_features}

        # Convert to DataFrame
        input_data = pd.DataFrame([user_input])

        # Make prediction
        predicted_label = rf.predict(input_data)[0]

        return render_template("index.html", features=selected_features, result=predicted_label)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

