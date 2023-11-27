#multi-class perceptron implementation

#import deepcopy
import copy
import numpy as np

class PMC:
    def __init__(self, n_classes, n_features, learning_rate=0.01):
        self.n_classes = n_classes
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.early_stopper = 3
        self.converging = 0
        self.converging_threshold = 0.02
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
        #compare weights to check if converged
        total_diff = 0.
        for i in range(len(self.weights)):
            total_diff += np.sum(np.abs(self.weights[i] - previous_weights[i]))
        if total_diff < self.converging_threshold:
            self.converging += 1
        else:
            self.converging = 0
        weights_equals = self.converging >= self.early_stopper
        return predictions, weights_equals
        
    def predict(self, X):
        scores = np.dot(self.weights, X)
        return np.argmax(scores)+1

    def score(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == y[i]:
                correct += 1
        return correct / len(X)