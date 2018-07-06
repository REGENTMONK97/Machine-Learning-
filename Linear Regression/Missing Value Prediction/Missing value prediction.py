import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_csv("D:\SAI RAGHUNATH.K\Softwares\Python\Machine Learning\LCSV files for assignment\Linear4.csv")
l = len(data["test"].values)
train_x = []
train_y = []
test_x = []
test_y = []
for i in range(l):
    if data["test"].values[i] == 0:
        x = data["number_of_seeds"].values[i]
        train_x.append(x)
        y = data["number_of_fruit"].values[i]
        train_y.append(y)
    if data["test"].values[i] == 1:
        y1 = data["number_of_fruit"].values[i]
        test_y.append(y1)
    if data["test"].values[i] == 2:
        x1 = data["number_of_seeds"].values[i]
        test_x.append(x1)

print(train_x)
print(train_y)
print(test_x)
print(test_y)
#x = np.reshape(x,(-1,1))
#y = np.reshape(y,(-1,1))

#model = LinearRegression()


