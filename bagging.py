import numpy as np
#np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn.neural_network as nn
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
# Load the data

data = pd.read_csv('iris.txt', header=None, sep='\t')

# Split the data into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
X_train = [resample(X_train, n_samples=len(X_train)//2, replace=True) for i in range(5)]
y_train = [resample(y_train, n_samples=len(y_train)//2, replace=True) for i in range(5)]


mlp = nn.MLPClassifier(
    hidden_layer_sizes=(3), activation='relu', solver='adam', random_state=2,
    learning_rate='adaptive', learning_rate_init=0.01, max_iter=2000)
class Bagging:
    def __init__(self, k = 5):
        self.k = k
        self.models = [nn.MLPClassifier(hidden_layer_sizes=(3), learning_rate='adaptive', learning_rate_init = 0.01, random_state=2, max_iter=2000) for i in range(k)]
    
    def train(self, X, y):
        for i in range(self.k):
            self.models[i].fit(X[i], y[i])
            
    def predict(self, X):
        predictions = []
        for i in range(self.k):
            predictions.append(self.models[i].predict(X))
        
        return np.array(predictions).T

    def choose_prediction(self, X):
        predictions = self.predict(X)
        return np.array([np.argmax(np.bincount(predictions[i])) for i in range(len(predictions))])
    

bagging = Bagging()
bagging.train(X_train, y_train)
test_predictions = bagging.choose_prediction(X_test)
print("============ Test Performance ============")

print(precision_score(y_test, test_predictions, average='macro'))
print(recall_score(y_test, test_predictions, average='macro'))
print(accuracy_score(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))
    

