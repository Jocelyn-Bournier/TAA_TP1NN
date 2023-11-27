import sklearn.neural_network as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
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

#normalize data using standard scaler
scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# Split the data into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#train perceptron until convergence
mlp = nn.MLPClassifier(
    hidden_layer_sizes=(3), activation='relu', solver='adam', random_state=2,
    learning_rate='adaptive', learning_rate_init=0.01, max_iter=2000)

# Train the model
mlp.fit(X_train, y_train)

print("============ Train Performance ============")
predictions = mlp.predict(X_train)
print(precision_score(y_train, predictions, average='macro'))
print(recall_score(y_train, predictions, average='macro'))
print(accuracy_score(y_train, predictions))
print(confusion_matrix(y_train, predictions))


print("============ Test Performance ============")
test_predictions = mlp.predict(X_test)
print(precision_score(y_test, test_predictions, average='macro'))
print(recall_score(y_test, test_predictions, average='macro'))
print(accuracy_score(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))
