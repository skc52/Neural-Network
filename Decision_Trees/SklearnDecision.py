from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Example data
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])
y = np.array([0, 0, 1, 1, 1])

# Initialize and train the DecisionTreeClassifier from sklearn
clf_sklearn = DecisionTreeClassifier(max_depth=2)
clf_sklearn.fit(X, y)

# Predict using the trained model
predictions_sklearn = clf_sklearn.predict(X)

# Output the predictions
print("Predictions:", predictions_sklearn)
