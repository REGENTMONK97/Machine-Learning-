from sklearn import datasets
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
diabetes_x = diabetes.data[:,np.newaxis,2]
#print(diabetes.target[-20:])

train_x = diabetes_x[:-20]
train_y = diabetes.target[:-20]

test_x = diabetes_x[-20:]
test_y = diabetes.target[-20:]

#train_x = np.reshape(train_x,(-1,1))
#train_y = np.reshape(train_y,(-1,1))
#test_x = np.reshape(test_x,(-1,1))
#test_y = np.reshape(test_y,(-1,1))

model = LinearRegression()

model_fit = model.fit(train_x,train_y)
score = model.score(train_x,train_y)
print(score)
y_predict = model.predict(test_x)
coeffecient = model.coef_
intercept = model.intercept_

points = [intercept+coeffecient[0]*i[0] for i in train_x]
plt.plot(points,"g*")
print(y_predict)
plt.plot(len(train_x)+y_predict[-20:],y_predict[-20:], color = "orange")
plt.show()

#x_train = data["x"][:-5].values
#y_train = data["y"][:-5].values

#x_test = data["x"][-5:].values
#y_test = data["y"][-5:].values

#print(x_test)
#print(y_test)

print(mean_squared_error(test_y,y_predict))
print(r2_score(test_y,y_predict))
