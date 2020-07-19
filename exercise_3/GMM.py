# 文件功能：实现 GMM 算法

import numpy as np
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.W = None
        self.pi = None
        self.Mu = None
        self.Var = None
    
    # 屏蔽开始
    # 更新W       # shape of W: (N, n_clusters)
    def update_W(self, X, pi, Mu, Var):
        W = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            W[:,i] = pi[i] * multivariate_normal.pdf(X, Mu[i], Var[i,:,:])
        W = W / np.sum(W, axis=1).reshape(-1,1)
        return W

    # 更新pi      shape of pi: (n_clusters, )
    def update_pi(self, X, W):
        N_k = np.sum(W, axis=0)
        pi = N_k / X.shape[0]
        return pi

    # 更新Mu      shape of Mu: (n_clusters, D)
    def update_Mu(self, X, W):
        Mu = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            Mu[i] = np.average(X, axis=0, weights=W[:,i])
        return Mu


    # 更新Var     shape of Var: (n_clusters, D, D)
    def update_Var(self, X, W, Mu):
        Var = np.zeros((self.n_clusters, X.shape[1], X.shape[1]))
        for i in range(self.n_clusters):
            Var[i,:,:] = np.dot((X-Mu[i]).T, np.dot(np.diag(W[:,i]),X-Mu[i]))
            Var[i,:,:] = Var[i,:,:]/np.sum(W[:,i])
        return Var

    # 更新log-likelihood
    def log_update(self, X, pi, Mu, Var):
        log = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            log[:,i] = pi[i] * multivariate_normal.pdf(X, Mu[i], Var[i,:,:])
        return np.sum(np.log(np.sum(log, axis=1)))



    # 屏蔽结束
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始
        N, dim = data.shape

        # 使用kmeans++初始化中心
        centroids = data[np.random.choice(np.arange(N), size=1, replace=False)]
        # for the remaining centroids, select by prob based on minimum distance to existing centroids:
        for _ in range(1, self.n_clusters):
            # find minimum distance to existing centroids for each point
            distances = np.asarray(
                [
                    np.min(np.linalg.norm(d - centroids, axis=1)) ** 2 for d in data
                ]
            )
            # generate cumulative probability:
            probs = distances / np.sum(distances)
            # print("probs: ", probs)
            cum_probs = np.cumsum(probs)
            # select new centroid:
            centroids = np.vstack(
                (centroids, data[np.searchsorted(cum_probs, random.random())])
            )

        Mu = centroids
        Var = np.zeros((self.n_clusters, dim, dim))
        for i in range(self.n_clusters):
            Var[i,:,:] = np.diag(np.ones(dim))
        pi = np.ones(self.n_clusters)/self.n_clusters
        log = 100

        for i in range(self.max_iter):
            W = self.update_W(data, pi, Mu, Var)
            Mu = self.update_Mu(data, W)
            Var = self.update_Var(data, W, Mu)
            pi = self.update_pi(data, W)

            log_new = self.log_update(data, pi, Mu, Var)
            if math.fabs(log_new-log) < 1e-3:
                break
            else:
                log = log_new

        self.W = W
        self.pi = pi
        self.Mu = Mu
        self.Var = Var

        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        W_pred = self.update_W(data, self.pi, self.Mu, self.Var)
        result = np.argmax(W_pred, axis=1)
        return result


        # 屏蔽结束

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

