from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = [[1,2,3],[9,8,7],[4,5,6],[10,11,12],[19,18,17],[14,15,16]]
model = KMeans(n_clusters = 2).fit(data)
print(model.score(data))
print(model.labels_)
print(model.cluster_centers_)
plt.plot(data,"b*")
plt.plot(model.cluster_centers_,"ro")
plt.show()