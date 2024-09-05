import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs, self.tree) for inputs in X]

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Check if the node is pure or max depth is reached
        if len(unique_classes) == 1:
            return unique_classes[0]
        if depth == self.max_depth:
            return self._most_common_class(y)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return self._most_common_class(y)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_branch = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_branch = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_branch, right_branch)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                gini = self._calculate_gini(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gini(self, left_y, right_y):
        left_size = len(left_y)
        right_size = len(right_y)
        total_size = left_size + right_size

        left_gini = 1 - sum((np.sum(left_y == c) / left_size) ** 2 for c in np.unique(left_y))
        right_gini = 1 - sum((np.sum(right_y == c) / right_size) ** 2 for c in np.unique(right_y))

        weighted_gini = (left_size / total_size) * left_gini + (right_size / total_size) * right_gini
        return weighted_gini

    def _most_common_class(self, y):
        return np.bincount(y).argmax()

    def _predict(self, inputs, node):
        if not isinstance(node, tuple):
            return node

        feature, threshold, left_branch, right_branch = node

        if inputs[feature] <= threshold:
            return self._predict(inputs, left_branch)
        else:
            return self._predict(inputs, right_branch)

# Example Usage
if __name__ == "__main__":
    # Example data
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6]
    ])
    y = np.array([0, 0, 1, 1, 1])

    # Initialize and train the decision tree
    clf = DecisionTree(max_depth=2)
    clf.fit(X, y)

    # Predict
    predictions = clf.predict(X)
    print("Predictions:", predictions)
