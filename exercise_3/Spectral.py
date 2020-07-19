import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

np.random.seed(10)
from KMeans import K_Means

class Spectral(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters


    def get_DistanceMatrix(self, X):
        N = X.shape[0]
        distance_matrix = np.zeros((N,N))
        for i in range(N-1):
            dist = np.linalg.norm(X[i+1:,:]-X[i], axis=1)
            distance_matrix[i,i+1:] = dist
        distance_matrix = distance_matrix + distance_matrix.T
        return distance_matrix


    def Distance_to_Weigt_knn(self, distance_matrix, k, sigma=1.0):
        N = distance_matrix.shape[0]
        factor =  1 / 2 / sigma / sigma
        adjacent = np.zeros((N,N))

        for i in range(N):
            dist = distance_matrix[i]
            idx_sorted = np.argsort(dist)
            adjacent[i,idx_sorted[:k]] = np.exp(distance_matrix[i,idx_sorted[:k]]*factor)
            adjacent[idx_sorted[:k],i] = np.exp(distance_matrix[i,idx_sorted[:k]]*factor)
        return adjacent


    def get_LaplacianMatrix(self, adjacent):
        degree_matrix = np.diag(np.sum(adjacent,axis=0))
        L = degree_matrix - adjacent
        L_rw = np.dot(np.linalg.inv(degree_matrix), L)
        return L_rw

    def get_YMatrix(self, laplacian_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
        idx_sorted = np.argsort(eigenvalues)

        Y = eigenvectors[:,idx_sorted][:,:self.n_clusters]
        return Y

    def fit(self, data):
        distance_matrix = self.get_DistanceMatrix(data)
        adjacent = self.Distance_to_Weigt_knn(distance_matrix, k=10)
        laplacian_matrix = self.get_LaplacianMatrix(adjacent)
        Y = self.get_YMatrix(laplacian_matrix)

        my_kmeans = K_Means(n_clusters=self.n_clusters)
        my_kmeans.fit(Y)
        self.labels = my_kmeans.predict(Y)

    def predict(self, data):
        return self.labels

    def plot(self, data, labels_my):
        # draw points according to predict cluster
        colors = ['red', 'green', 'blue', 'black']
        plt.figure(figsize=(10, 8))
        for i in range(self.n_clusters):
            x = data[labels_my == i]
            plt.scatter(x[:, 0], x[:, 1], c=colors[i], s=5)
        plt.show()



if __name__ == '__main__':
    data, label_gt = datasets.make_circles(n_samples=500, factor=0.5, noise=0.05)

    spc = Spectral(n_clusters=2)
    spc.fit(data)
    result = spc.predict(data)
    spc.plot(data, result)
