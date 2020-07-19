import os
import numpy as np
import open3d as o3d
import pandas as pd
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

def main():
    # root directory of dataset
    root_dir = "../../../dataset/modelnet40_normal_resampled"
    point_cloud_files = read_data(root_dir)

    for i, file in enumerate(point_cloud_files):
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

        r = 0.05      # voxel filter size

        x_max = np.max(X[:,0])
        x_min = np.min(X[:,0])

        y_max = np.max(X[:,1])
        y_min = np.min(X[:,1])

        z_max = np.max(X[:,2])
        z_min = np.min(X[:,2])

        d_x = np.ceil((x_max-x_min)/r)
        d_y = np.ceil((y_max-y_min)/r)
        d_z = np.ceil((z_max-z_min)/r)

        index = np.floor((X[:,0]-x_min)/r) + np.floor((X[:,1]-y_min)/r)*d_x + np.floor((X[:,2]-z_min)/r)*d_x*d_y

        df = pd.DataFrame(X)
        df.columns = ["x", "y", "z"]
        df["index"] = index

        voxel_exact = df.groupby(['index']).mean().to_numpy()
        groupped = df.groupby(['index'])
        voxel_random = []
        for name, group in groupped:
            sample = group.sample(1)[['x','y','z']].to_numpy()
            sample = sample.reshape(3)
            voxel_random.append(sample)

        voxel_random = np.array(voxel_random, dtype=float)

        # voxel_random = df.groupby(['index']).apply(
        #     lambda x: x[['x', 'y', 'z']].sample(1)
        # ).to_numpy()




        pcd_exact = o3d.geometry.PointCloud()
        pcd_exact.points = o3d.utility.Vector3dVector(voxel_exact)
        o3d.visualization.draw_geometries([pcd_exact],window_name="exact voxel filter for object {}".format(i))

        pcd_random = o3d.geometry.PointCloud()
        pcd_random.points = o3d.utility.Vector3dVector(voxel_random)
        o3d.visualization.draw_geometries([pcd_random],window_name="random voxel filter for object {}".format(i))

if __name__ == '__main__':
    main()