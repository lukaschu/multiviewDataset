import numpy as np
import matplotlib as plt
import cv2
import io,os
import open3d as o3d
import pickle

from camera_params import CameraIntrinsics

class Depth_To_Pointcloud():

    def __init__(self,file_path):
        # Directory witht the depth images
        self.path = file_path
        self.cam_list = ['840412062035','840412062037','840412062038','840412062076']
        self.CameraIntrinsics = CameraIntrinsics
        

    # idx: correpsonds to picture sequence
    # iv: corresponds to camera {0,1,2,3} 
    def readRGB(self,idx,iv):
        rgbpath = os.path.join(self.path, 'rgb')
        rgbpath = os.path.join(rgbpath, "%05d" % (idx) + '_' + str(iv) + '.jpg')
        return cv2.imread(rgbpath)

    # idx: correpsonds to picture sequence
    # iv: corresponds to camera {0,1,2,3}
    def readDepth(self,idx,iv):
        depthpath = os.path.join(self.path, 'depth')
        depthpath = os.path.join(depthpath, "%05d" % (idx) + '_' + str(iv) + '.png')
        depth_image = cv2.imread(depthpath)

        # Decoding the rgb image into the real depth image (depth given in milimeters)
        r, g, _ = depth_image[:, :, 0], depth_image[:, :, 1], depth_image[:, :, 2]
        depth = (r.astype(np.uint64) + g.astype(np.uint64) * 256).astype(np.uint16)
        return depth
    
    # Returnes the mask that segments the hand from the scene (inaccurate!!)
    def readMask(self,idx,iv):
        maskpath = os.path.join(self.path, 'mask')
        maskpath = os.path.join(maskpath, "%05d" % (idx) + '_' + str(iv) + '.jpg')
        return cv2.imread(maskpath)

    # Gnerates a single point cloud of a single picture
    def Generate_Pointcloud(self,idx,iv):
        # Initialize depth image
        depth = self.readDepth(idx,iv)
        mask = self.readMask(idx,iv)
        mask = mask[:,:,0]/255 ## all are the same

        # Lay mask over depth image
        depth = mask * depth

        #introduce internal params, self.cam_list[iv] returns string name of the iv camera
        fx = self.CameraIntrinsics[self.cam_list[iv]].fx 
        fy = self.CameraIntrinsics[self.cam_list[iv]].fy
        cx= self.CameraIntrinsics[self.cam_list[iv]].cx
        cy= self.CameraIntrinsics[self.cam_list[iv]].cy

        point_cloud = []
        for i in range(depth.shape[0]):
            for k in range(depth.shape[1]):
                z = depth[i,k]
                x = z * (k-cx) / fx
                y = z * (i-cy) / fy
                # Only consider 3d points that are relevant
                if (z > 0):
                    point = [x,y,z]
                    point_cloud.append(point)

        return point_cloud

    def plot_some_stuff(self,idx,iv, image):

        cv2.imshow('image',image)
        cv2.waitKey(0)
        print(f"Image resolution: {image.shape}")
        print(f"Data type: {image.dtype}")
        print(f"Min value: {np.min(image)}")
        print(f"Max value: {np.max(image)}")




