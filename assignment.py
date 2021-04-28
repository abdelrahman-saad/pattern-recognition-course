import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


def calculateMSE(X,Y,theta):
    Y_pred = np.dot(X,theta)
    error = Y_pred - Y
    squareError = np.square(error)
    sumSquareError = np.sum(squareError)
    dim = Y.shape[0]
    dim *= 2 #Multiplying by half
    MSE = sumSquareError / dim
    return MSE


def calculateGradient(X,Y,theta):
    yPredict = np.dot(X,theta)
    error = yPredict - Y
    xTranspose = X.transpose()
    gradient = np.dot(xTranspose,error)
    return gradient

data = pd.read_csv('Fish.csv')
X = data[['Length1','Length2','Length3','Height','Width']]
X_bias = np.ones((X.shape[0], 1)) # 159
X_new = np.append(X_bias, X, axis=1)

Y = data['Weight']
X_new = np.array(X_new)

Y = np.array(Y)

Y = np.reshape(Y,(Y.shape[0],1)) # (159,1)

theta = np.zeros((6,1))
sc = StandardScaler()
X_new = sc.fit_transform(X_new)
maxIter = 100
tolerance = 0.0000000001
alpha = 0.01
errors = []
iterations = []
for i in range(maxIter):
    curCost = calculateMSE(X_new,Y,theta)
    if (i+1) % 10 == 0:
        errors.append(curCost)
        iterations.append(i)
    if curCost < tolerance:
        break
    gradient = calculateGradient(X_new,Y,theta)
    theta = theta - alpha* (gradient/Y.shape[0])

print(errors)
plt.plot(iterations,errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Gradient Descent')
plt.show()
