import numpy as np
from sklearn.datasets import load_breast_cancer
from my_linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

data = load_breast_cancer()
x = data.data
y = data.target

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.05, shuffle=True)
print(f'Training set shape:\n\txtrain: {xtrain.shape}\n\tytrain: {ytrain.shape}')
print(f'Testing set shape:\n\txtest: {xtest.shape}\n\tytest: {ytest.shape}')

epoch = 10_000
clf = LogisticRegression(max_iter=epoch)
clf.fit(xtrain, ytrain)
ypred = clf.predict(xtest)

print(confusion_matrix(ytest, ypred))
print('Accuracy Score: {:.3f}%'.format(accuracy_score(ytest, ypred)*100))
