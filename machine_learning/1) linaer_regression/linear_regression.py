import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
#import sklearn
from sklearn.linear_model import LinearRegression  
df = pd.read_csv("/home/factious/Desktop/machine_learning/linaer_regression/linear-regression-dataset.csv")
# plt.scatter(df.deneyim,df.maas)
# plt.xlabel("deneyim")
# plt.ylabel("maas")
# plt.show()
#sklearn
linear_reg = LinearRegression()
x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)
linear_reg.fit(x,y)
b0_=linear_reg.intercept_
print(b0_)
b0=linear_reg.predict([[0]])
print(b0)
b1 = linear_reg.coef_
print(b1)

#maas= b0+b1*maas
#21 yillik deneyim icin ornek maas
m21=b0+(b1*21)
print(m21)
print(linear_reg.predict([[11]]))
print(linear_reg.predict([[21]]))
L = list(range(1,16))
array = np.array(L).reshape(-1,1)
plt.scatter(x,y)
y_head = linear_reg.predict(array)
plt.plot(array,y_head,color='red')
plt.show()
print(linear_reg.predict([[100]]))
print(df.columns)