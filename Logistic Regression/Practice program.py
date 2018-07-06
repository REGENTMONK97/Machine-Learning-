import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("D:\SAI RAGHUNATH.K\Softwares\Python\Machine Learning\LCSV files for assignment\Linear1.csv")

train_x = data["x"][:-2].values.reshape(-1,1)
train_y = data["y"][:-2].values.reshape(-1,1)

test_x = np.ravel(data["x"][-2:].values).reshape(-1,1)
test_y = np.ravel(data["y"][-2:].values).reshape(-1,1)
model = LogisticRegression()
model.fit(train_x,np.ravel(train_y.astype(int)))
score = model.score(train_x,np.ravel(train_y.astype(int)))
print(score)
intercept = model.intercept_
coeff = model.coef_


predict_y = model.predict(test_x)
print("prediction:",predict_y)
#plt.plot(len(train_x),predict_y,"bo")

def logistic_model(x):
    return(1/(1+np.exp(-x)))
points =[intercept+coeff[0]*i[0] for i in (train_x)]
points = np.ravel([logistic_model(i) for i in points])
plt.plot(points)

