#multi-class perceptron implementation

#import deepcopy
import copy
import numpy as np

class PMC:
    def __init__(self, n_classes, n_features, learning_rate=0.1):
        self.n_classes = n_classes
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.weights = np.random.rand(n_classes, n_features)

    def fit(self, X, y):
        previous_weights = self.weights.copy()
        predictions = []
        for i in range(len(X)):
            prediction = self.predict(X[i])
            predictions.append(prediction)
            if prediction != y[i]:
                self.weights[prediction-1] -= self.learning_rate * X[i]
                self.weights[y[i]-1] += self.learning_rate * X[i]
        return predictions, np.array_equal(previous_weights, self.weights)
        
    def predict(self, X):
        scores = np.dot(self.weights, X)
        return np.argmax(scores)+1

    def score(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == y[i]:
                correct += 1
        return correct / len(X)