import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load the trained model, selected features, and personality mapping
rf = joblib.load("random_forest_model.pkl")
selected_features = joblib.load("selected_features.pkl")

# MBTI Personality Type Mapping
personality_map = {
    0: "ENFJ", 1: "ENFP", 2: "ENTJ", 3: "ENTP",
    4: "ESFJ", 5: "ESFP", 6: "ESTJ", 7: "ESTP",
    8: "INFJ", 9: "INFP", 10: "INTJ", 11: "INTP",
    12: "ISFJ", 13: "ISFP", 14: "ISTJ", 15: "ISTP"
}

# Initialize Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", features=selected_features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs from form
        user_input = {feature: float(request.form[feature]) for feature in selected_features}

        # Convert to DataFrame
        input_data = pd.DataFrame([user_input])

        # Make prediction
        predicted_label = rf.predict(input_data)[0]
        predicted_personality = personality_map.get(predicted_label, "Unknown Personality")

        return render_template("index.html", features=selected_features, result=predicted_personality)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

