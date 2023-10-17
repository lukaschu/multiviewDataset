from extraction import Depth_To_Pointcloud
import numpy as np
import matplotlib as plt
import cv2
import io,os
import open3d as o3d
import pickle
import math


class Full_Pointcloud_Maker(Depth_To_Pointcloud):

    def __init__(self,file_path,idx):
        # Object containing the single image point cloud extractor function
        estimator = Depth_To_Pointcloud(file_path)

        # Contains all four point clouds of the same scene
        self.Four_Pointclouds = []
        for iv in range(4):
            self.Four_Pointclouds.append(estimator.Generate_Pointcloud(idx,iv))

        # Extract the homogeneous Transf. w.r.t the fourth image coordinate system
        file_path = os.path.join(file_path, 'calib.pkl')
        with open(file_path, 'rb') as f:
            self.camera_pose_map = pickle.load(f)

    # Get homogeneous transformation
    # Note that the transf betweeen the four images is contained in the calib.pkl file!
    def Get_Transform(self,iv):
        cam_list = ['840412062035','840412062037','840412062038','840412062076']
        return self.camera_pose_map[cam_list[iv]]

    # Transforms a pointcloud to the frame of the fourth image
    def PointCloud_Transform(self,iv):
        partial_cloud = np.array(self.Four_Pointclouds[iv])

        # Extract homog. transformation
        Transform = self.Get_Transform(iv)

        # Now we perform the whole transformation
        n = partial_cloud.shape[0]
        partial_cloud = np.concatenate((partial_cloud,np.ones((n,1))),axis=1)
        NEW_partial_cloud = (Transform @ partial_cloud.T).T

        # Back to euclidean space
        return np.array(NEW_partial_cloud[:,0:3])

    # Complete the full pointcloud by adding them together
    def Point_Cloud_completion(self):
        # Last point cloud is ref. frame
        Full_Point_Cloud = np.array(self.Four_Pointclouds[3])
        for iv in range(3):
            Partial_pointcloud = self.PointCloud_Transform(iv)
            Full_Point_Cloud = np.concatenate((Full_Point_Cloud, Partial_pointcloud), axis = 0)
        return Full_Point_Cloud
    
    # Pointcloud smoothing /filtering
        

if __name__ == "__main__":
    print("Starting extraction process")
    file_path = "/Users/lukasschuepp/framework/hand_data/data/7-14-1-2"

    # idx is frame
    idx = 0
    heyooo = Full_Pointcloud_Maker(file_path,idx)
    Complete = heyooo.Point_Cloud_completion()
    Complete /= 1000
    
    # Visualize
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    
    pcd_o3d.points = o3d.utility.Vector3dVector(Complete)  # set pcd_np as the point cloud points
    
    # Outlier removal with statistical approach
    #pcd_stat, ind_stat = pcd_o3d.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.6)

    # Outliers removal with radial approach
    #pcd_rad, ind_r = pcd_o3d.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.6)

    # Cropping the image
    #min_bound = [-math.inf, -math.inf, 0.3]
    #max_bound = [math.inf, math.inf, 0.9]
    #crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    #pcd_o3d = pcd_o3d.crop(crop_box)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    draw_geometries = [pcd_o3d, coord]
    o3d.visualization.draw_geometries(draw_geometries)
    
