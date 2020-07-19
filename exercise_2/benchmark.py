import random
import math
import numpy as np
import time
import os
import struct
import open3d as o3d
from scipy import spatial

import kd_tree as kd_tree
import octree as octree
from result_set import KNNResultSet, RadiusNNResultSet



def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def brute_force(db_np, indices, k, N):
    print("using brute force...")
    result = []
    for i in range(N):
        query = db_np[i, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        point_indices = indices[nn_idx][:k]
        result.append(point_indices)
    return np.array(result, dtype=int)


def scipy_kd_tree(db_np, k, leaf_size,N):
    print("using scipy kd-Tree...")
    tree = spatial.KDTree(db_np, leafsize=leaf_size)
    d,  point_indices= tree.query(db_np[:N, :], k, p=2)
    return point_indices



def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1
    N = 30000    # 对N个点做搜索

    # read data
    filename = '../000000.bin'
    db_np = read_velodyne_bin(filename)
    indices = np.arange(db_np.shape[0])

    # brute force
    begin_t = time.time()
    brute_result_indices = brute_force(db_np, indices, k, N)
    brute_time = time.time()-begin_t
    print("time consumption for brute force: ", brute_time)


    # scipy kd-tree
    begin_t = time.time()
    scipy_result_indices = scipy_kd_tree(db_np, k, leaf_size, N)
    scipy_time = time.time() - begin_t
    print("time consumption for scipy kd-tree: ", scipy_time)



    # kd tree
    begin_t = time.time()
    print("constructing a kd-tree")
    root = kd_tree.kd_tree_construction(db_np, leaf_size)
    kd_tree_construction_time = time.time() - begin_t
    print("construct a kd-tree takes ", kd_tree_construction_time)

    begin_t = time.time()
    print("using kd-tree-knn")
    kd_tree_knn_result = []
    for i in range(N):
        knn_result_set = KNNResultSet(k)
        kd_tree.kd_tree_knn_search(root, db_np, knn_result_set, db_np[i])
        kd_tree_knn_result.append(knn_result_set)
    kd_tree_knn_time = time.time() - begin_t
    print("time consumption for kd-tree-knn: ", kd_tree_knn_time)


    begin_t = time.time()
    print("using kd-tree-radius")
    kd_tree_radius_result = []
    for i in range(N):
        radius_result_set = RadiusNNResultSet(radius)
        kd_tree.kd_tree_radius_search(root, db_np, radius_result_set, db_np[i])
        kd_tree_radius_result.append(radius_result_set)
    kd_tree_radius_time = time.time() - begin_t
    print("time consumption for kd-tree-radius: ", kd_tree_radius_time)




    # octree
    begin_t = time.time()
    print("constructing a octree")
    root = octree.octree_construction(db_np, leaf_size, min_extent)
    octree_construction_time = time.time() - begin_t
    print("construct a octree takes ", octree_construction_time)

    begin_t = time.time()
    print("using octree-knn")
    octree_knn_result = []
    for i in range(N):
        knn_result_set = KNNResultSet(k)
        octree.octree_knn_search(root, db_np, knn_result_set, db_np[i])
        octree_knn_result.append(knn_result_set)
    octree_knn_time = time.time() - begin_t
    print("time consumption for octree-knn: ", octree_knn_time)

    begin_t = time.time()
    print("using octree-radius")
    octree_radius_result = []
    for i in range(N):
        radius_result_set = RadiusNNResultSet(radius)
        octree.octree_radius_search(root, db_np, radius_result_set, db_np[i])
        octree_radius_result.append(radius_result_set)
    octree_radius_time = time.time() - begin_t
    print("time consumption for octree-radius: ", octree_radius_time)





    # print the reuslt
    print("---------------- brute force reusult ---------------------: ")
    print(brute_result_indices[2])
    print("---------------- scipy kd-tree-knn reusult ----------------: ")
    print(scipy_result_indices[2])
    print("----------------- kd-tree-knn reusult ---------------------: ")
    print(kd_tree_knn_result[2])
    print("------------------ kd-tree-radius reusult -----------------: ")
    print(kd_tree_radius_result[2])
    print("------------------ octree-knn reusult ---------------------: ")
    print(octree_knn_result[2])
    print("------------------ octree-radius reusult ------------------: ")
    print(octree_radius_result[2])



if __name__ == '__main__':
    main()