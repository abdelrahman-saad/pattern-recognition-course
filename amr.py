import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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
max_iterations = 100
Alfa = 0.01
tolerance = 0.0000000001
theta=np.zeros((6,1))
sc = StandardScaler()
X_new = sc.fit_transform(X_new)
Errors=[]
for i in range (max_iterations):
    Current_Cost = (np.sum(0.5*np.abs(pow((np.dot(X_new,theta)-Y),2))))/Y.shape[0]
    if (i+1)%10==0:
        Errors.append(Current_Cost)
    if Current_Cost< tolerance:
        break

    Grad = np.dot(X_new.transpose(),(np.dot(X_new,theta)-Y))
    Gradient = Grad/Y.shape[0]
    theta = theta-Alfa*Gradient
print(Errors)
plt.plot(Errors)
plt.show()