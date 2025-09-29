import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(r"C:\Users\beere\OneDrive\Desktop\disease_symptom.csv")

# Features and target
X = data.drop("disease", axis=1)
y = data["disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate (optional)
y_pred = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))

# ---- Prediction ----
print("\nEnter your symptoms (1 for yes, 0 for no):")
input_data = []
for symptom in X.columns:
    val = int(input(f"Do you have {symptom}? (1/0): "))
    input_data.append(val)

# Predict disease
input_df = pd.DataFrame([input_data], columns=X.columns)
prediction = model.predict(input_df)[0]

print("\n🩺 Predicted Disease:", prediction)
