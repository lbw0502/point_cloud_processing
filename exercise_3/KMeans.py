# 文件功能： 实现 K-Means 算法

import numpy as np
import random

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        N,D = data.shape


        # 选任意点为初始化中心
        # centroid_idxs = np.random.choice(np.arange(N), size=self.k_, replace=False)
        # centroids = data[centroid_idxs]


        # # 使用kmeans++初始化中心
        centroids = data[np.random.choice(np.arange(N), size=1, replace=False)]

        # for the remaining centroids, select by prob based on minimum distance to existing centroids:
        for _ in range(1, self.k_):
            # find minimum distance to existing centroids for each poit
            distances = np.asarray(
                [
                    np.min(np.linalg.norm(d - centroids, axis=1)) ** 2 for d in data
                ]
            )
            # print("distances: ", distances)
            # generate cumulative probability:
            probs = distances / np.sum(distances)
            # print("probs: ", probs)
            cum_probs = np.cumsum(probs)
            # print("cum_probs: ", cum_probs)
            # select new centroid:
            centroids = np.vstack(
                (centroids, data[np.searchsorted(cum_probs, random.random())])
            )



        centroids_new = np.zeros_like(centroids)

        for itr in range(self.max_iter_):
            point_groups = [[] for k in range(self.k_)]
            dist = np.zeros((N, self.k_))
            for i in range(self.k_):
                dist[:,i] = np.linalg.norm(data-centroids[i], axis=1)
            indices = np.argmin(dist, axis=1)
            for i in range(N):
                point_groups[indices[i]].append(data[i])

            for i in range(self.k_):
                centroids_new[i] = np.mean(np.array(point_groups[i]), axis=0)

            self.centroids = centroids_new
            diff = centroids_new - centroids
            if np.sum(np.linalg.norm(diff, axis=1)) < self.tolerance_:
                break
            else:
                centroids = centroids_new


        # 屏蔽结束

    def predict(self, p_datas):
        # 作业2
        # 屏蔽开始
        dist = np.zeros((p_datas.shape[0], self.k_))
        for i in range(self.k_):
            dist[:, i] = np.linalg.norm(p_datas - self.centroids[i], axis=1)
        result = np.argmin(dist, axis=1)

        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

