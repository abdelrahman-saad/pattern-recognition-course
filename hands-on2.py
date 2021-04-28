import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# read data
data = pd.read_csv('Fish.csv')
data_encoded = pd.DataFrame()

#To Do: apply OneHotEncoder on 'Species' column
onehotencoder = preprocessing.OneHotEncoder()
data_encoded = onehotencoder.fit_transform(data['Species'].values.reshape(-1, 1)).toarray()
df_encoded = pd.DataFrame(data_encoded, columns = ["Species_"+str(int(i)) for i in range(data.shape[1])])
data_encoded = pd.concat([data, df_encoded], axis=1)
#To Do: put your data in data_encoded variable

# Select columns for  X , Y
data_encoded= data_encoded.drop(['Species'], axis=1) 
X = data_encoded.drop(['Weight'], axis=1).values
Y = data_encoded['Weight'].values.reshape(-1, 1)

#To Do: Normalize your X, using MinMax Scaler
norm = MinMaxScaler().fit(X)
X = norm.transform(X)
#To Do: fit LinearRegression
model=LinearRegression(fit_intercept=True)
model.fit(X,Y)

#To Do: get predictions
Y_pred = model.predict(X)

Error = sum((Y_pred - Y)**2) / (2 * Y.shape[0])
print(Error)
