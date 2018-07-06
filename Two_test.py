import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("D:\SAI RAGHUNATH.K\Softwares\Python\Machine Learning\linear.csv")

train_x = data["x"][:-5].values
train_y = data["y"][:-5].values
test_x = data["x"][-5:].values
test_y = data["y"][-5:-3].values

train_x = np.reshape(train_x,(-1,1))
train_y = np.reshape(train_y,(-1,1))
test_x = np.reshape(test_x,(-1,1))
test_y = np.reshape(test_y,(-1,1))

model = LinearRegression()
model.fit(train_x,train_y)
score = model.score(train_x,train_y)
print(score)
coefficient = model.coef_
intercept = model.intercept_
point = [intercept+coefficient[0]*i[0] for i in train_x]
plt.plot(point,"bo")
prediction = model.predict(test_x)
print("prediction ",prediction)
l = len(prediction)
plt.plot(len(data)+prediction,prediction,"g*")
#plt.plot(len(data),prediction[0],"g*")
#plt.plot(len(data)+len(prediction[0]),prediction[1],"b*")

print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
plt.show()