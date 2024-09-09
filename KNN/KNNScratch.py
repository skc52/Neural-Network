import numpy as np
from collections import Counter

# Function to calculate the Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN Class
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    # Fitting the data (in KNN, this is just storing the training data)
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
    
    # Predict a class label for each point in X_test
    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = [self._predict_single_point(x) for x in X_test]
        return np.array(predictions)
    
    # Helper function to predict a single point's class
    def _predict_single_point(self, x):
        # Calculate distances between x and all points in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get the indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k-nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote, the most common class label among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage
if __name__ == "__main__":
    # Sample training data (features and labels)
    X_train = [[2, 3], [1, 1], [4, 4], [3, 2], [3, 3], [7, 7]]
    y_train = [0, 0, 1, 1, 1, 1]
    
    # New test data points (to classify)
    X_test = [[1, 2], [5, 5], [3, 3], [7, 6]]

    # Initialize the KNN model with k=3
    knn = KNN(k=3)
    
    # Fit the model (store the training data)
    knn.fit(X_train, y_train)
    
    # Predict the class for the test points
    predictions = knn.predict(X_test)
    
    print("Predictions:", predictions)
