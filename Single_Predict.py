#import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
#import matplotlib.pyplot as plt

n = int(input("Enter the number of elements in each list: "))
li1 = [];li2 = []
print("Enter list 1")
for i in range(n):
    a = int(input())
    li1.append(a)
print("Enter list 2")
for i in range(n):
    a = int(input())
    li2.append(a)
#m = input("Enter the metric: ")

train_x = li1[:-1]
train_y = li2[:-1]

train_x = np.reshape(train_x,(-1,1))
train_y = np.reshape(train_y,(-1,1))

test_x = li1[-1:]
test_y = li2[-1:]
test_x = np.reshape(test_x,(-1,1))
test_y = np.reshape(test_y,(-1,1))

model = LinearRegression()
model.fit(train_x,train_y)
score = model.score(train_x,train_y)
print(score)
coefficient = model.coef_
intercept = model.intercept_
point = [intercept+coefficient[0]*i[0] for i in train_x]
#plt.plot(point,"bo")
prediction = model.predict(test_x)
print("prediction ",prediction)
l = len(prediction)
#plt.plot(len(data),prediction[0],"b*")
#plt.plot(len(data)+len(prediction[0]),prediction[1],"b*")

print(mean_squared_error(test_y,prediction))
print(r2_score(test_y,prediction))
#plt.show()