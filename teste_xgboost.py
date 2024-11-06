import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def softmax(x):
    # Compute softmax values for each set of scores in x.
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def predict_classes(model, data, num_classes):
    # Convert input data to DMatrix format
    dmatrix = xgb.DMatrix(data)
    # Get raw scores
    raw_scores = model.get_booster().predict(dmatrix, output_margin=True).reshape(-1, num_classes)
    # Apply softmax
    probabilities = softmax(raw_scores)
    # Get class with the highest probability
    return np.argmax(probabilities, axis=1), probabilities

# Load Iris data
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBClassifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Get the Booster
booster = model.get_booster()

# Extract and print each tree
for i, tree in enumerate(booster.get_dump()):
    print(f"Tree {i}:\n{tree}\n")

# Usage
labels, unique = pd.factorize(y)
predicted_classes, class_probabilities = predict_classes(model, X_test, len(unique))

print("Predicted Classes:", predicted_classes)
print("Class Probabilities:\n", class_probabilities)