from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (features and labels)
X = [[2, 3], [1, 1], [4, 4], [3, 2], [3, 3], [7, 7], [8, 8], [5, 5], [2, 1], [6, 6]]
y = [0, 0, 1, 1, 1, 1, 1, 1, 0, 1]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Predicted labels:", y_pred)
print("Actual labels:", y_test)
print("Accuracy:", accuracy)
