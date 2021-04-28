import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculateTheta(X,Y):
    X_transpose = X.transpose()
    X_temp = np.dot(X_transpose,X)
    X_inverse = np.linalg.inv(X_temp)
    otherTemp = np.dot(X_transpose,Y)
    theta = np.dot(X_inverse,otherTemp)
    return theta

def calculateMSE(X,Y,theta):
    Y_pred = np.dot(X,theta)
    error = Y_pred - Y
    squareError = np.square(error)
    sumSquareError = np.sum(squareError)
    dim = Y.shape[0]
    dim *= 2 #Multiplying by half
    MSE = sumSquareError / dim
    return MSE
    
# read data
data = pd.read_csv('Fish.csv')
# Select columns for  X , Y
X = data[['Length1','Length2','Length3','Height','Width']]
# Add bias vector with all ones.
X_bias = np.ones((X.shape[0], 1)) # 159
X_new = np.append(X_bias, X, axis=1)

Y = data['Weight']
# create Numpy arrays from X , Y
X_new = np.array(X_new)

Y = np.array(Y)
Y = np.reshape(Y,(159,1)) # (159,1)
# X : shape = (159 , 6) , Y : shape = (159 , 1)
# theta should be of shape (6,1)
# TO DO : Apply Normal Equation to find theta then use theta to predict
theta = calculateTheta(X_new,Y)
# To DO : Use theta to predict the label Y
#Y_pred = np.dot(X_new,theta)

#To DO : calculate Error (loss) between prediction and Y
#mse = np.divide(sumSquareError,dim) Can't because it's a vector
MSE = calculateMSE(X_new,Y,theta)
print(MSE)