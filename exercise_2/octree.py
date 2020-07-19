import numpy as np
import math
import time
import os
import struct

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


class Octant:
    def __init__(self, center, extent, point_indices, child):
        self.center = center
        self.extent = extent
        self.point_indices = point_indices
        self.child = child
        self.is_leaf = False


def traverse_octree(root: Octant, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root is None:
        pass
    elif root.is_leaf:
        pass
    else:
        for child in root.child:
            traverse_octree(child, depth, max_depth)
    depth[0] -= 1


def octree_construction(db_np, leaf_size, min_extent):
    N, dim = db_np.shape
    point_indices = np.arange(N)

    db_min = np.amin(db_np, axis=0)
    db_max = np.amax(db_np, axis=0)
    extent = np.max(db_max-db_min)*0.5
    center = db_min+extent

    root = None
    root = octree_construction_core(root, db_np, point_indices, center, extent, leaf_size, min_extent)

    return root


def octree_construction_core(root, db_np, point_indices, center, extent, leaf_size, min_extent):
    if root is None:
        root = Octant(center, extent, point_indices, [None for i in range(8)])

    if len(point_indices)<=leaf_size or extent<=min_extent:
        root.is_leaf = True
        return root

    else:
        root.is_leaf = False
        child_indices = [[] for i in range(8)]
        for point_index in point_indices:
            point = db_np[point_index]
            morton_code = 0
            if point[0] > center[0]:
                morton_code = morton_code | 1
            if point[1] > center[1]:
                morton_code = morton_code | 2
            if point[2] > center[2]:
                morton_code = morton_code | 4
            child_indices[morton_code].append(point_index)

        factor = [-0.5, 0.5]
        for i in range(8):
            child_center_x = center[0] + factor[(i & 1) > 0] * extent
            child_center_y = center[1] + factor[(i & 2) > 0] * extent
            child_center_z = center[2] + factor[(i & 4) > 0] * extent

            child_center = np.asarray([child_center_x, child_center_y, child_center_z])
            child_extent = extent*0.5

            root.child[i] = octree_construction_core(root.child[i], db_np, child_indices[i], child_center, child_extent, leaf_size, min_extent)

    return root



def overlap(query, radius, octant):

    query_offset = np.fabs(query - octant.center)
    max_dist = radius + octant.extent
    if np.any(query_offset>max_dist):
        return False

    if np.sum((query_offset < octant.extent).astype(np.int)) >= 2:
        return True

    x_diff = max(query_offset[0] - octant.extent, 0)
    y_diff = max(query_offset[1] - octant.extent, 0)
    z_diff = max(query_offset[2] - octant.extent, 0)

    return x_diff*x_diff + y_diff*y_diff + z_diff*z_diff > radius*radius


def inside(query, radius, octant):
    query_offset = np.fabs(query - octant.center)
    possibile_space = query_offset + radius
    return np.all(octant.extent > possibile_space)



def octree_knn_search(root, db_np, knn_result_set, query):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices)>0:
        data = db_np[root.point_indices]
        diff = np.linalg.norm(np.expand_dims(query, 0) - data, axis=1)
        for i in range(len(diff)):
            knn_result_set.add_point(diff[i], root.point_indices[i])

    else:
        morton_code = 0
        if query[0] > root.center[0]:
            morton_code = morton_code | 1
        if query[1] > root.center[1]:
            morton_code = morton_code | 2
        if query[2] > root.center[2]:
            morton_code = morton_code | 4

        if octree_knn_search(root.child[morton_code], db_np, knn_result_set, query):
            return True
        else:
            for i in range(8):
                if i == morton_code or root.child[i] is None:
                    continue
                if overlap(query, knn_result_set.worstDist(), root.child[i]) is False:
                    continue
                if octree_knn_search(root.child[i], db_np, knn_result_set, query):
                    return True

    return inside(query, knn_result_set.worstDist(), root)



def octree_radius_search(root, db_np, radius_result_set, query):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices)>0:
        data = db_np[root.point_indices]
        diff = np.linalg.norm(np.expand_dims(query, 0) - data, axis=1)
        for i in range(len(diff)):
            radius_result_set.add_point(diff[i], root.point_indices[i])

    else:
        morton_code = 0
        if query[0] > root.center[0]:
            morton_code = morton_code | 1
        if query[1] > root.center[1]:
            morton_code = morton_code | 2
        if query[2] > root.center[2]:
            morton_code = morton_code | 4

        if octree_knn_search(root.child[morton_code], db_np, radius_result_set, query):
            return True
        else:
            for i in range(8):
                if i == morton_code or root.child[i] is None:
                    continue
                if overlap(query, radius_result_set.worstDist(), root.child[i]) is False:
                    continue
                if octree_knn_search(root.child[i], db_np, radius_result_set, query):
                    return True

    return inside(query, radius_result_set.worstDist(), root)







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


    root = octree_construction(db_np, leaf_size, min_extent)

    depth = [0]
    max_depth = [0]
    traverse_octree(root, depth, max_depth)

    query = db_np[0]
    radius_result_set = RadiusNNResultSet(radius)
    octree_radius_search(root, db_np, radius_result_set, query)
    print(radius_result_set)

    # knn_result_set = KNNResultSet(k)
    # octree_radius_search(root, db_np, knn_result_set, query)
    # print(knn_result_set)
    #






if __name__ == '__main__':
    main()