import os
import numpy as np
import open3d as o3d
import random


def read_data(root_dir):

    dirs = os.listdir(root_dir)

    point_cloud_path = []

    for dir in dirs:
        path = os.path.join(root_dir, dir)
        if os.path.isdir(path):
            files = os.listdir(path)
            num = random.randint(0, len(files)-1)
            full_path = os.path.join(path, files[num])
            point_cloud_path.append(full_path)

    return point_cloud_path


def PCA(data, correlation=False, sort=True):

    mean = np.mean(data, 0)
    data = data - mean

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(data.T,data))

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def pca_o3d_computation(data, eigenvectors):
    mean = np.mean(data,0)
    points = np.row_stack((eigenvectors.T, np.zeros(3)))
    points = points + mean
    lines = [
        [0, 3],
        [1, 3],
        [2, 3],
    ]

    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def normals_o3d_compution(data,k):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = []
    N = data.shape[0]
    for i in range(N):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(data[i, :], k)
        surface = data[idx]
        w, v = PCA(surface)
        normals.append(v[:, 2])
    normals = np.array(normals, dtype=float)

    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd




def main():
    # root directory of dataset
    root_dir = "../../../dataset/modelnet40_normal_resampled"
    point_cloud_files = read_data(root_dir)

    for file in point_cloud_files:

        f = open(file,'r')
        line = f.readline()
        points = np.fromstring(line,dtype='float', sep=',')
        points = points.reshape(1,6)

        while line:
            line = f.readline()
            row = np.fromstring(line, dtype='float', sep=',')
            if row.size>0:
                points = np.row_stack((points,row))
        f.close()




        X = points[:,0:3]
        N = X.shape[0]      # size of X: n*3

        # task1
        w, v = PCA(X)
        pca_o3d = pca_o3d_computation(X, v)

        # task2
        normals_o3d = normals_o3d_compution(X, 20)

        o3d.visualization.draw_geometries([pca_o3d, normals_o3d],window_name=file,point_show_normal=True)

if __name__ == '__main__':
    main()