#importing modules

import numpy as np
n = int(input("Enter the number of elements in each list: "))
li1 = [];li2 = []
print("Enter list 1")
for i in range(n):
    a = input()
    li1.append(a)
print("Enter list 2")
for i in range(n):
    a = input()
    li2.append(a)
m = input("Enter the metric: ")

x_train = li1[:-1]
y_train = li2[:-1]
print(x_train)
print(y_train)
x_train = np.reshape(x_train,(-1,1))
y_train = np.reshape(y_train,(-1,1))

x_test = li1[-1:]
y_test = li2[-1:]
x_test = np.reshape(x_test,(-1,1))
y_test = np.reshape(y_test,(-1,1))

l = len(x_train)

xs = 0
ys = 0
for i in range(l):
    xs =xs + int(x_train[i])
x_avg = xs/l
for i in y_train:
    ys = ys+int(i)
y_avg = ys/l
xvs = 0 # variance sum
for i in x_train:
    xvs = xvs+((int(i)-x_avg)**2)
b11 = 0
for i in range(l):
    b11 = b11 + ((int(x_train[i])-x_avg)*(int(y_train[i])-y_avg))

b1 = b11/xvs          #b1 is obtained
b0 = y_avg - b1*x_avg #b0 is obtained

y_predict = []

for i in x_test:
    p = b0+b1*int(i)
    y_predict.append(p)

print(y_predict)

if m == "RMSE":
    rm = 0
    for i in range(n-l):
        rm = rm + (1/(n-l)*((int(y_predict[i])-int(y_test[i]))**2))
    rmse = rm**(1/2)
    print("RMSE:",rmse)
if m == "MSE":
    mse = 0
    for i in range(n - l):
        mse = mse + (1 / (n - l) * ((int(y_predict[i]) - int(y_test[i])) ** 2))
    print("mse:",mse)
if m == "R2":
    r2 = 0
    yvs = 0  # variance sum
    for i in y_train:
        yvs = yvs + ((int(i) - y_avg) ** 2)
    ypvs = 0
    for i in range(n-l):
        ypvs = ypvs+((int(y_test[i])-int(y_predict[i]))**2)
    print("R2:",ypvs)