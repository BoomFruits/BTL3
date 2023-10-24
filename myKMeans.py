import numpy as np
import matplotlib.pyplot as plt
class KMeans:
    def __init__(self,k = 3):
        self.k = k
        self.center = np.array([[2,10],[2,5],[8,4]])

    @staticmethod
    def eclid_distance(data_point,center):
        return np.sqrt(np.sum((center-data_point)**2,axis = 1)) 
    def fit(self,X,max_iter = 200):
        self.center = np.random.uniform(np.amin(X,axis=0),np.amax(X,axis=0),
                                        size=(self.k,X.shape[1]))
        for _ in range(max_iter):
            y = [] 
            for data_point in X:
                distances = KMeans.eclid_distance(data_point,self.center)
                cluster_num = np.argmin(distances) #return the index of the smallest values
                y.append(cluster_num)
            y = np.array(y)
            cluster_indices = []
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))
            cluster_centers = []
            for i,indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.center[i])
                else:
                    cluster_centers.append(np.mean(X[indices],axis=0))
            if np.max(self.center - np.array(cluster_centers) < 1e5):
                break
            else:
                self.center = np.array(cluster_centers)
        return y
X = np.array([[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9],[2,8]])
kmeans = KMeans(k=3)
labels = kmeans.fit(X)
print(labels)
