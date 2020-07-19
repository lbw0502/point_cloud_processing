import numpy as np
import struct
import os
import math

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



class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

def next_axis(axis, dim):
    if axis == dim-1:
        return 0
    else:
        return axis+1


def traverse_kdtree(root, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        pass
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1



def kd_tree_construction(db_np, leaf_size):
    N, dim = db_np.shape
    root = None
    root = kd_tree_construction_core(root, db_np, np.arange(N), leaf_size=leaf_size, axis = 0)
    return root

def kd_tree_construction_core(root, db_np, point_indices, leaf_size, axis):
    if root is None:
        root = Node(axis=axis, value=None, left=None, right=None, point_indices=point_indices)

    if len(point_indices) > leaf_size:
        # point data from the corresponding axis
        data = db_np[point_indices][:, axis]
        idxs_sorted = np.argsort(data)
        data_sorted = data[idxs_sorted]

        mid = math.floor(len(data_sorted)/2)+1
        left_idxs = idxs_sorted[:mid]
        left_points_indicies = point_indices[left_idxs]
        right_idxs = idxs_sorted[mid:]
        right_points_indicies = point_indices[right_idxs]

        root.value = (data_sorted[mid-1] + data_sorted[mid-1])/2

        root.left = kd_tree_construction_core(root.left, db_np, left_points_indicies, leaf_size, next_axis(axis,db_np.shape[1]))
        root.right = kd_tree_construction_core(root.right, db_np, right_points_indicies, leaf_size, next_axis(axis,db_np.shape[1]))

    return root

def kd_tree_knn_search(root, db_np, knn_result_set, query):
    if root is None:
        return False

    if root.is_leaf() is True:
        data = db_np[root.point_indices]
        diff = np.linalg.norm(np.expand_dims(query, 0) - data, axis=1)
        for i in range(len(diff)):
            knn_result_set.add_point(diff[i], root.point_indices[i])
        return False

    else:
        if query[root.axis] < root.value:
            kd_tree_knn_search(root.left, db_np, knn_result_set, query)
            if math.fabs(query[root.axis]-root.value) < knn_result_set.worstDist():
                kd_tree_knn_search(root.right, db_np, knn_result_set, query)

        else:
            kd_tree_knn_search(root.right, db_np, knn_result_set, query)
            if math.fabs(query[root.axis] - root.value) < knn_result_set.worstDist():
                kd_tree_knn_search(root.left, db_np, knn_result_set, query)

    return False


def kd_tree_radius_search(root, db_np, radius_result_set, query):

    if root is None:
        return False

    if root.is_leaf() is True:
        data = db_np[root.point_indices]
        diff = np.linalg.norm(np.expand_dims(query, 0) - data, axis=1)
        for i in range(len(diff)):
            radius_result_set.add_point(diff[i], root.point_indices[i])
        return False

    else:
        if query[root.axis] < root.value:
            kd_tree_knn_search(root.left, db_np, radius_result_set, query)
            if math.fabs(query[root.axis]-root.value) < radius_result_set.worstDist():
                kd_tree_knn_search(root.right, db_np, radius_result_set, query)

        else:
            kd_tree_knn_search(root.right, db_np, radius_result_set, query)
            if math.fabs(query[root.axis] - root.value) < radius_result_set.worstDist():
                kd_tree_knn_search(root.left, db_np, radius_result_set, query)

    return False











def main():

    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1
    N = 100000    # 对N个点做搜索

    # read data
    filename = "../000000.bin"
    db_np = read_velodyne_bin(filename)

    root = kd_tree_construction(db_np, leaf_size)

    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)

    knn_result_set = KNNResultSet(k)
    query = db_np[0,:]
    kd_tree_knn_search(root, db_np, knn_result_set, query)
    print(knn_result_set)

    radius_result_set = RadiusNNResultSet(radius)
    kd_tree_radius_search(root, db_np, radius_result_set, query)
    print(radius_result_set)


    





if __name__ == "__main__":
    main()