import torch
import torch.utils.data as data
from segmentation.data_utils.SemanticKittiDataset import load_kitti_label_map
from Constants.constants import ROOT_DIR
import os
from pathlib import Path
import numpy as np
import open3d as o3d
"""import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D"""

class P2Net_Dataset(data.Dataset):
    def __init__(self, npoints=2048, split='train', experimental=True, num_seq = 3):
        self.data_root = os.path.join(ROOT_DIR, 'data', 'SemanticKitti')
        self.npoints = npoints
        self.split = split.strip().lower()
        self.experimental = experimental
        # maps raw labels to sequential labels for learning
        self.learning_map = load_kitti_label_map('learning_map')
        # maps the sequential learning labels back to categorical label indices
        self.inv_map = load_kitti_label_map('learning_map_inv')
        # maps the actual categorical label indices to names
        self.name_map = load_kitti_label_map('labels')
        # How many sequence of a input (current + previous)
        self.num_seq = num_seq

        # specific scenes are used for train,val, and test respectively
        # we do not have access to test labels, but we need to output test results and submit to competition site
        # if the experimental flag is set, we use different splits so we have labels for all data
        scenes = []
        if self.split == 'train':
            if self.experimental:
                scenes = ['00', '01', '02', '03', '06', '07', '09', '10']
            else:
                scenes = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            if self.experimental:
                scenes = ['08']
            else:
                scenes = ['08']
        elif self.split == 'test':
            if self.experimental:
                scenes = ['04', '05']
            else:
                scenes = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

        self.file_list = []
        self.index_list = []
        for scene in scenes:
            match_path = os.path.join(self.data_root, 'SemanticKitti', 'dataset', 'sequences', scene)
            bin_list = list(
                Path.glob(Path(match_path), os.path.join('*', '*.bin'))
            )
            length = len(bin_list)
            for i in range(length):
                if i >= num_seq - 1:
                    self.index_list.append([scene, str(i).zfill(6)])

        """# for each velodyne point cloud frame, we check if there is a label
        for i, point_file in enumerate(self.file_list):
            # XXXXXX from XXXXXX.bin
            frame_number = point_file.stem
            # frame_number = os.path.basename(point_file)[-4]
            # grabbing label path
            label_assumed_path = point_file.parent.parent.joinpath(Path('labels', frame_number + '.label'))
            # label_assumed_path = point_file.joinpath(Path('..','labels',str(frame_number),'.label'))
            if label_assumed_path.exists():
                # add the label as a tuple
                self.file_list[i] = (point_file, label_assumed_path)
            else:
                # set the label to None
                self.file_list[i] = (point_file, None)"""

    def __getitem__(self, idx):
        # construct the file path using id of scene and frame
        scene, frame = self.index_list[idx]
        current_point_cloud_file = os.path.join(self.data_root, 'SemanticKitti', 'dataset', 'sequences', scene, 'velodyne', frame + '.bin')
        current_label_file = os.path.join(self.data_root, 'SemanticKitti', 'dataset', 'sequences', scene, 'labels', frame + '.label')

        current_point_cloud = np.fromfile(current_point_cloud_file, dtype=np.float32).reshape((-1, 4))[:, :3]

        # read the previous point cloud
        previous_point_cloud = []
        for i in range(1, self.num_seq):
            temp_frame = str(int(frame) - i).zfill(6)
            temp_point_cloud_file = os.path.join(self.data_root, 'SemanticKitti', 'dataset', 'sequences', scene,  'velodyne',
                                                    temp_frame + '.bin')
            previous_point_cloud.append(np.fromfile(temp_point_cloud_file, dtype=np.float32).reshape((-1, 4))[:, :3])

        # aligning the previous point clouds to the current one
        previous_point_cloud = align(current_point_cloud, previous_point_cloud)

        return None

    def __len__(self):
        return len(self.index_list)

def get_pcd_from_numpy(pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    return pcd

def find_transformation(source, target, trans_init):
    threshold = 0.2
    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    transformation = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                       o3d.pipelines.registration.TransformationEstimationPointToPlane()).transformation
    return transformation
def align(current, previous_list):
    # Convert numpy point clouds to Open3D PointCloud objects
    pcd_source = get_pcd_from_numpy(current)
    T = None
    aligned_previous_list = []

    for i in range(len(previous_list)):
        if T is None:
            pcd_tmp = get_pcd_from_numpy(previous_list[i])
            T = find_transformation(pcd_tmp, pcd_source, np.eye(4))
            aligned_pcd = np.asarray(pcd_tmp.transform(T).points)
            aligned_previous_list.append(aligned_pcd)
        else:
            pcd_tmp = get_pcd_from_numpy(previous_list[i])
            T = find_transformation(pcd_tmp, get_pcd_from_numpy(previous_list[i-1]), np.eye(4)) @ T
            aligned_pcd = np.asarray(pcd_tmp.transform(T).points)
            aligned_previous_list.append(aligned_pcd)

    return aligned_previous_list

if __name__ == '__main__':
    dataset = P2Net_Dataset()
    print(dataset[0])