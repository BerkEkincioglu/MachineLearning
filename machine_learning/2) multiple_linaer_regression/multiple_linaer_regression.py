import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

df= pd.read_csv("/home/factious/Desktop/machine_learning/multiple_linaer_regression/multiple_linear_regression.csv")

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)
b0 = multiple_linear_regression.intercept_
print("b0: ",b0)
print("b1,b2 : ",multiple_linear_regression.coef_)

print(multiple_linear_regression.predict(np.array([[10,35],[5,35]])))