import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier,kneighbors_graph
df = pd.read_csv("D:\SAI RAGHUNATH.K\Softwares\Python\Machine Learning\Balance.csv",header=None)
# print(df)
from sklearn.linear_model import LogisticRegressionCV

df.columns = ['ColA','ColB','ColC','ColD','ColE']
df.rename(columns={'ColA': 'cn', 'ColB': 'lw', 'ColC': 'ld', 'ColD': 'rw', 'ColE': 'rd'},inplace=True)
X = df[['lw','ld','rw','rd']][:-2].values
Y = df["cn"][:-2].values

x_test = df[['lw','ld','rw','rd']][-2:].values
y_test = df["cn"][-2:].values

print("KNN----------------")
model1 = KNeighborsClassifier(n_neighbors=3)

model1.fit(X,Y.ravel())

score = model1.score(X,Y.ravel())
print(score)

print(x_test, y_test)
print(model1.predict(x_test))

print("This should predict : left-L -> 5,5,5,3")
print(model1.predict([[5,5,5,3]]))

print("This should predict : Both-B -> 3,5,5,3")
print(model1.predict([[3,5,5,3]]))

print("This should predict : Right-R -> 3,5,5,5")
print(model1.predict([[3,5,5,5]]))

# print("Doing some feature engineering ======================================")

df['newcol1'] = df['lw']*df['ld']
df['newcol2'] = df['rw']*df['rd']

X = df[['newcol1','newcol2']][:-3].values
Y = df["cn"][:-3].values

x_test = df[['newcol1','newcol2']][-3:].values
y_test = df["cn"][-3:].values

print("KNN----------------")
model1 = KNeighborsClassifier(n_neighbors=3)

model1.fit(X,Y.ravel())

score = model1.score(X,Y.ravel())
print(score)

print(x_test, y_test)
print(model1.predict(x_test))

print("This should predict : left-L -> 5,5,5,3")
print(model1.predict([[(5*5),(5*3)]]))

print("This should predict : Both-B -> 3,5,5,3")
print(model1.predict([[(3*5),(5*3)]]))

print("This should predict : Right-R -> 3,5,5,5")
print(model1.predict([[(3*5),(5*5)]]))

# print("LOGR---------------")
#
model2 = LogisticRegressionCV(cv=10,multi_class = "multinomial")

model2.fit(X,Y)

score = model2.score(X,Y)
print(score)

print(x_test, y_test)
print(model2.predict(x_test))

print("This should predict : left-L -> 5,5,5,3")
print(model1.predict([[(5*5),(5*3)]])) # works fine
print(model1.predict([[(1.5*5),(1.5*4)]])) # works fine
print(model1.predict([[(20*15),(22*5)]])) # works fine

print("This should predict : Both-B -> 3,5,5,3")
print(model1.predict([[(3*5),(5*3)]])) # works fine
print(model1.predict([[(2.2*3),(3.3*2)]])) # works fine
print(model1.predict([[(6*6),(6*6)]])) # doesn't work
print(model1.predict([[(3*20),(20*3)]])) # doesn't work

print("This should predict : Right-R -> 3,5,5,5")
print(model1.predict([[(3*5),(5*5)]])) # works fine
print(model1.predict([[(5.1*1.5),(3.5*2.9)]]))  # works fine
print(model1.predict([[(31*5),(20*5)]])) # works fine