import numpy as np
import os
import open3d as o3d

def readMetadata(file):
    M = np.zeros((4, 4))
    with open(file) as f:
        timeStamp = int(f.readline())
        for i in range(4):
            M[i, :] = np.array(list(map(lambda x: float(x),f.readline().strip().split(", "))))
    M = M.T
    # M[:3, 3] *= 1000
    M = np.linalg.inv(M)
    return M

def normalize_pcd(pcd):

    pcd_array = np.array(pcd.points)

    # centralize data
    pcd_centralized = pcd_array - np.mean(pcd_array, axis=0)

    # normalize data
    m = np.max(np.sqrt(np.sum(pcd_centralized**2, axis=1)))
    pcd_normalized = pcd_centralized / m

    pcd.points = o3d.utility.Vector3dVector(pcd_normalized)
    return pcd


def visualize_depth(ply_file, M, ply_raw):
    # read .ply file
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.array(pcd.points) # * 1000
    points[:, 2] *= -1
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.transform(M)

    origin = np.array([0, 0, 0, 1])
    # add a coordinate frame
    mesh_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # remove points around the origin from pcd with threshold
    threshold = 0.45
    points = np.asarray(pcd.points)
    dist = np.linalg.norm(points - origin[:3], axis=1)
    mask = dist > threshold
    pcd = pcd.select_by_index(np.where(mask)[0])

    # create a visualization window
    o3d.visualization.draw_geometries([pcd, mesh_origin])


    pcd_normalized = normalize_pcd(pcd)

    # save pcd as .pcd file
    save_path = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/HoloLens/Recording_29_11_23/'
    o3d.io.write_point_cloud(os.path.join(save_path, frame + '_filtered.pcd'), pcd_normalized, write_ascii=True)




if __name__ == "__main__":
    base_path = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/HoloLens/Recording_29_11_23/depth/'
    segmentation_path = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/HoloLens/colored_ply_one_hand/'
    raw_path = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/HoloLens/Recording_29_11_23/depth/'
    frame = '000290'
    file = os.path.join(base_path, 'meta_' + frame + '.txt')
    ply_raw = os.path.join(raw_path, frame + '.ply')
    M = readMetadata(file)
    ply_file = os.path.join(segmentation_path, frame + '.ply')
    visualize_depth(ply_file, M, ply_raw)
