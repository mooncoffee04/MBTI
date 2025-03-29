import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Change this to your actual dataset

# Encode target variable (Personality)
label_encoder = LabelEncoder()
df["Personality_Encoded"] = label_encoder.fit_transform(df["Personality"])

# Separate features and target
X = df.drop(columns=["Response Id", "Personality", "Personality_Encoded"], errors='ignore')
y = df["Personality_Encoded"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the best parameters from your tuning
best_params = {
    "n_estimators": 150,
    "max_depth": 17,
    "min_samples_split": 6,
    "min_samples_leaf": 2,
    "random_state": 42
}

# Train Random Forest model
rf = RandomForestClassifier(**best_params)
rf.fit(X_train, y_train)

# Feature Selection using RFE
num_features_to_select = 10  
rfe = RFE(estimator=rf, n_features_to_select=num_features_to_select)
rfe.fit(X_train, y_train)
selected_features = X.columns[rfe.support_]

# Personality Mapping
personality_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", features=selected_features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs
        user_input = {feature: float(request.form[feature]) for feature in selected_features}

        # Convert to DataFrame
        input_data = pd.DataFrame([user_input])

        # Predict personality
        predicted_label = rf.predict(input_data)[0]
        predicted_personality = personality_map.get(predicted_label, "Unknown Personality")

        return render_template("index.html", features=selected_features, result=predicted_personality)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)


