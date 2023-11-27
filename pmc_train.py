import numpy as np
#np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import PMC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('iris.txt', header=None, sep='\t')

#normalize data using standard scaler
scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# Split the data into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Plot the data and labels
#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#train perceptron until convergence
pmc = PMC.PMC(3, 4)
converged = False
max_iters = 5000
iters = 0
predictions = []

while not converged and iters < max_iters :
    predictions, converged = pmc.fit(X_train, y_train)
    iters += 1
print("Converged in " + str(iters) + " iterations")

print("============ Train Performance ============")
print(precision_score(y_train, predictions, average='macro'))
print(recall_score(y_train, predictions, average='macro'))
print(accuracy_score(y_train, predictions))
print(confusion_matrix(y_train, predictions))

print("============ Test Performance ============")
test_predictions = [pmc.predict(x) for x in X_test]
print(precision_score(y_test, test_predictions, average='macro'))
print(recall_score(y_test, test_predictions, average='macro'))
print(accuracy_score(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))