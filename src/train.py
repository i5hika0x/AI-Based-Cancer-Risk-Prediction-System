import os
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load real dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Parameters
params = {
    'model__n_estimators': [100],
    'model__max_depth': [None, 10]
}

# Train model
grid = GridSearchCV(pipeline, params, cv=5)
grid.fit(X, y)

print("Accuracy:", grid.best_score_)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)

print("✅ Model trained and saved!")