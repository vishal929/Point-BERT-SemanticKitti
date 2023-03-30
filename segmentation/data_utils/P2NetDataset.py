import torch
import torch.utils.data as data
from segmentation.data_utils.SemanticKittiDataset import load_kitti_label_map, SemanticKitti
import functools
import os
import segmentation.provider as provider
import torch.nn as nn
import numpy as np
import open3d as o3d
from knn_cuda import KNN

"""import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D"""

class P2Net_Dataset(SemanticKitti):
    def __init__(self, npoints=2048, split='train', experimental=True, num_seq=3):
        super().__init__(npoints=npoints, split=split, experimental=experimental)

        # num_seq = 1 + num of previous time step
        self.num_seq = num_seq

    def __getitem__(self, idx):
        # construct the file path using id of scene and frame
        point_cloud_file, label_file = self.file_list[idx]

        # get the scene and frame id from the path
        scene, frame = get_seq_frame(point_cloud_file)

        # to make sure the frame have n-1 previous frames
        if int(frame) < self.num_seq - 1:
            frame = str(self.num_seq - 1).zfill(6)

        point_clouds = []
        point_clouds_labels = None
        for i in range(0, self.num_seq):
            temp_frame = str(int(frame) - i).zfill(6)

            # get the file path using the scene and the frame id
            temp_point_cloud_file = os.path.join(self.data_root, 'SemanticKitti', 'dataset', 'sequences', scene,
                                                 'velodyne', temp_frame + '.bin')
            temp_labels_file = os.path.join(self.data_root, 'SemanticKitti', 'dataset', 'sequences', scene,
                                                 'labels', temp_frame + '.label')

            # read point_cloud from file
            point_cloud = np.fromfile(temp_point_cloud_file, dtype=np.float32).reshape((-1, 4))

            # get the labels and filter out the outliers
            labels = None

            if label_file is not None:
                # lower 16 bits give the semantic label
                # higher 16 bits gives the instance label
                labels = np.fromfile(temp_labels_file, dtype=np.uint32).reshape((-1))
                labels = labels & 0xFFFF
                # we have 16 bits of unsigned information, we can represent this in int32
                labels = labels.astype(np.int32)
                # apply the mapping from raw labels to labels used for learning
                labels = np.vectorize(self.learning_map.get)(labels)
                # removing labels and points which correspond to 0 category (outlier) as it is not used for train or eval
                filter_map = ~(labels == 0)
                labels = labels[filter_map]
                point_cloud = point_cloud[filter_map]
                # IMPORTANT!!!!
                # decrement labels by 1 so labels 1-19 are related to cls-one-hot from 0-18
                labels -= 1
                # then we can just increment predictions by 1 and reverse map them

            point_cloud = point_cloud
            if labels is not None:
                labels = torch.from_numpy(labels)
            if point_clouds_labels is None:
                point_clouds_labels = labels

            point_clouds.append(point_cloud)

        # align the points to the current's scenes'
        point_clouds = align(point_clouds)
        if self.npoints is not None:
            flag = True
            for i, point_cloud in enumerate(point_clouds):
                # we sample from the point cloud, and if for some reason we sample less than npoints, we resample with replacement
                sampling_indices = np.random.choice(len(point_cloud), min(self.npoints, len(point_cloud)), replace=False)
                if len(sampling_indices) < self.npoints:
                    new_sample = np.random.choice(len(point_cloud), self.npoints-len(point_cloud), replace=True)
                    sampling_indices = np.concatenate([sampling_indices,new_sample])
                point_clouds[i] = point_cloud[sampling_indices, :]
                if flag:
                    point_clouds_labels = point_clouds_labels[sampling_indices]
                    flag = False

        return point_clouds, point_clouds_labels

    def __len__(self):
        return len(self.file_list)

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
def align(points_list):
    current = points_list[0]
    previous_list = points_list[1:]

    # Convert numpy point clouds to Open3D PointCloud objects
    pcd_source = get_pcd_from_numpy(current)
    T = None
    aligned_previous_list = [current]

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

    for i in range(len(points_list)):
        points_list[i][:, 0:3] = aligned_previous_list[i][:, 0:3]

    return points_list

def get_seq_frame(path):
    # Split the path into directory and filename
    dir_path, filename = os.path.split(path)

    # Get the parent directory name (i.e., '00') from the directory path
    parent_dir = os.path.basename(os.path.dirname(dir_path))

    # Extract the number from the filename
    number = filename.split('.')[0]

    return parent_dir, number

def col(item, model=None, device='cpu'):
    point_clouds = [tmp_item[0] for tmp_item in item]
    labels = [tmp_item[1] for tmp_item in item]

    batch_size = len(point_clouds)
    num_seq = len(point_clouds[0])
    num_points = len(point_clouds[0][0])

    point_clouds = [tmp_point for tmp_points in point_clouds for tmp_point in tmp_points]

    #-----------------------------------------------------------------Point Bert-------------------------------------------------
    # batch up the point_clouds
    points_pb= torch.stack([torch.Tensor(point_cloud) for point_cloud in point_clouds])[:, :, 0:3] # (batch_size * num_seq, num_points, 3)
    points_pb = points_pb.data.numpy()
    points_pb[:, :, 0:3] = provider.random_scale_point_cloud(points_pb[:, :, 0:3])
    points_pb[:, :, 0:3] = provider.shift_point_cloud(points_pb[:, :, 0:3])
    points_pb = torch.Tensor(points_pb)
    points_pb = points_pb.float().to(device)

    points_pb = points_pb.transpose(2, 1)
    seg_pred, _ = model(points_pb, None) #(batch_size * num_seq, num_points, cls_num)

    seg_pred = seg_pred.reshape(batch_size, num_seq, num_points, -1)
    seg_pred = seg_pred.permute(0, 2, 1, 3).contiguous()
    seg_pred = seg_pred.reshape(batch_size, num_points, -1)
    #-----------------------------------------------------------------Point Bert-------------------------------------------------







    return torch.ones(1)


# a fake model to test the dataset
class test_Model(nn.Module):
    def __init__(self, num_cls):
        super().__init__()
        self.num_cls = num_cls
    def forward(self, x,y):
        tensor_shape = np.array(x.shape)
        B, _, num_points = tensor_shape
        x = torch.randn((B, num_points, self.num_cls))
        return x, y

if __name__ == '__main__':
    fake_point_bert = test_Model(num_cls=19)

    dataset = P2Net_Dataset()

    collate_fn = functools.partial(col, model=fake_point_bert)

    loader = data.DataLoader(dataset, batch_size=2, collate_fn= collate_fn)

    for i, item in enumerate(loader):
        print(item)
