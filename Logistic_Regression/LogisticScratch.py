import math
import random 

from sklearn.linear_model import LogisticRegression

class LogisticRegressionScratch:
    def __init__(self, learning_rate = 0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None 
        self.bias = None

    def sigmoid(self, z):
        return 1/(1+math.exp(-z))
    

    def train(self, X, y):
        #initialize paramters

        num_samples, num_features = len(X), len(X[0])
        self.weights = [0.0 for _ in range(num_features)]
        self.bias = 0.0


        #Gradient Descent

        for _ in range(self.iterations):
            for i in range(num_samples):
                linear_model = sum([self.weights[j]*X[i][j] for j in range(num_features)]) + self.bias
                y_pred = self.sigmoid(linear_model)

                #Calculate gradients
                error = y_pred - y[i]

                for j in range(num_features):
                    self.weights[j] -= self.learning_rate*error*X[i][j]
                self.bias -= self.learning_rate*error


    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            linear_model = sum([self.weights[j]*X[i][j] for j in range(len(self.weights))])
            y_pred = self.sigmoid(linear_model)
            predictions.append(1 if y_pred>=0.5 else 0)
        return predictions
    
if __name__ == "__main__":
    # Example training data (X: features, y: labels)
    X = [[0.5, 1.5], [1.0, 2.0], [1.5, 0.5], [2.0, 3.0]]
    y = [0, 0, 1, 1]

     # Initialize and train the model
    model = LogisticRegressionScratch(learning_rate=0.1, iterations=1000)
    model.train(X, y)

    # Make predictions
    X_test = [[1.5, 2.5], [0.3, 0.5]]
    predictions = model.predict(X_test)
    print("Predictions:", predictions)


    # Initialize the model
    model2 = LogisticRegression()

    # Train the model
    model2.fit(X, y)

    # Make predictions
    predictions = model2.predict(X_test)

    print("Predictions 2:", predictions)


