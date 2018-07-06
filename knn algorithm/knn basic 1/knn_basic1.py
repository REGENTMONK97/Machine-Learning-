import numpy as np
from sklearn.neighbors import kneighbors_graph,KNeighborsClassifier
import pandas as pd

drinks = pd.read_csv("D:\SAI RAGHUNATH.K\KNNsample.csv")

train_x = drinks[["Sweetness","Fizziness"]].values
train_y = drinks["Type of Drink"].values

test_x = [[5,7]]
test_y = ["Cold Drink"]

model = KNeighborsClassifier(n_neighbors=2)

model.fit(train_x,train_y)
score = model.score(train_x,train_y)
print(score)

classify = model.predict(test_x)
print(classify)