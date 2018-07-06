from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
data = np.reshape(data,(-1,1))

model = KMeans(n_clusters = 2).fit(data)
print(model.cluster_centers_)
print(model.labels_)
score = model.score(data)
print(score)