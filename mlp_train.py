import sklearn.neural_network as nn
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
np.set_printoptions(threshold=10000,suppress=True)
import warnings
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('iris.txt', header=None, sep='\t')
classes = 3
features = 4

# Split the data into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#train perceptron until convergence
pmc = nn.MLPClassifier(
    hidden_layer_sizes=(3), activation='relu', solver='adam',
    learning_rate='adaptive', learning_rate_init=0.01, max_iter=2000)

# Train the model
pmc.fit(X_train, y_train)

# Make predictions
predictions = pmc.predict(X_test)

# calculate accuracy
print("Accuracy: ", metrics.accuracy_score(y_test, predictions))
