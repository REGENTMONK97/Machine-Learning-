import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("D:\SAI RAGHUNATH.K\Softwares\Python\Machine Learning\Multi_Features Assignment\data1.csv")

#train_x = pd.DataFrame(data,columns = data[["X","Y"]][:-2].values)
train_x = data[["X","Y"]][:-2].values.reshape(-1,2)
#train_x = np.reshape(train_x,(-1,2))
#print(train_x)
#train_x2 = data["Y"][:-2].values
#train_x2 = np.reshape(train_x2,(-1,1))
train_y = data["Expected_output"][:-2].values.reshape(-1,1)
#train_y = np.reshape(train_y,(-1,1))

#test_x = pd.DataFrame(data,columns = data[["X","Y"]][-2:].values)
test_x = data[["X","Y"]][-2:].values.reshape(-1,2)
#test_x = np.reshape(test_x,(-1,2))
#test_x2 = data["Y"][-2:].values
#test_x2 = np.reshape(test_x2,(-1,1))
test_y = data["Expected_output"][-2:].values.reshape(-1,1)
#test_y = np.reshape(test_y,(-1,1))

#print(test_x["X"])
model = LinearRegression()
model.fit(train_x,train_y)

coeff = model.coef_
intercept = model.intercept_
points = [intercept+(coeff[0]*i[0]) for i in train_x]
plt.plot(points,"ro")
predict_y = model.predict(test_x)
plt.plot(train_y,predict_y,"b*")

print(predict_y)
plt.show()
#intercept =
#points =
