import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import PMC

# Load the data
data = pd.read_csv('iris.txt', header=None, sep='\t')

# Split the data into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Plot the data and labels
#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train perceptron until convergence
pmc = PMC.PMC(3, 4)
converged = False
delta_precision = 0.05
iters_delta = 5
cpt_delta = 0
last_score = 0
score = 0
predictions = []

def recall_score(predictions, y):
    nCorrectClass = []
    nPerClass = np.zeros(3)
    for i in range(len(predictions)):
        nPerClass[y[i]] += 1
        if predictions[i] == y[i]:
            nCorrectClass[predictions[i]] += 1
    return [nCorrectClass[i]/nPerClass[i] for i in range(3)]

while not converged :
    predictions, converged = pmc.fit(X_train, y_train)
    score = pmc.score(X_train, y_train)
    if(abs(score - last_score) < delta_precision):
        cpt_delta += 1
    else:
        cpt_delta = 0
    last_score = score
    if not converged :
        converged = cpt_delta == iters_delta

print(pmc.score(X_train, y_train))
print(pmc.score(X_test, y_test))
print(recall_score(predictions, y_train))