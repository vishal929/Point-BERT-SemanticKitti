import torch
import torch.utils.data as data
from segmentation.data_utils.SemanticKittiDataset import load_kitti_label_map, SemanticKitti
import functools
import os
import segmentation.provider as provider
import torch.nn as nn
import numpy as np
import open3d as o3d
from pathlib import Path
#from knn_cuda import KNN
import faiss

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
        # we only align after sampling and getting raw predictions
        #point_clouds = align(point_clouds)
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
def get_seq_frame(path):
    # Split the path into directory and filename
    dir_path, filename = os.path.split(path)

    # Get the parent directory name (i.e., '00') from the directory path
    parent_dir = os.path.basename(os.path.dirname(dir_path))

    # Extract the number from the filename
    number = filename.split('.')[0]

    return parent_dir, number

# the collatn_fn function for loading the P2NetDataset into the pytorch dataloader
def P2Net_collatn(item, model=None, device='cuda'):
    point_clouds = [tmp_item[0] for tmp_item in item]
    if item[0][1] is not None:
        labels = [tmp_item[1] for tmp_item in item]
        labels = torch.stack(labels)

    batch_size = len(point_clouds)
    num_seq = len(point_clouds[0])
    num_points = len(point_clouds[0][0])

    point_clouds = [tmp_point for tmp_points in point_clouds for tmp_point in tmp_points]

    points_clouds = torch.stack([torch.Tensor(point_cloud) for point_cloud in point_clouds]).reshape(batch_size,num_seq,num_points,-1).float()

    # -----------------------------------------------------------------Point Bert-------------------------------------------------
    # batch up the point_clouds
    points_pb = points_clouds.reshape(-1, num_points, 4)[:, :, 0:3]  # (batch_size * num_seq, num_points, 3)
    points_pb = points_pb.numpy()
    points_pb[:, :, 0:3] = provider.random_scale_point_cloud(points_pb[:, :, 0:3])
    points_pb[:, :, 0:3] = provider.shift_point_cloud(points_pb[:, :, 0:3])
    points_pb = torch.Tensor(points_pb)
    points_pb = points_pb.float().to(device)

    # get the model predict for every point clouds
    points_pb = points_pb.transpose(2, 1)
    seg_pred, _ = model(points_pb, None)  # (batch_size * num_seq, num_points, cls_num)

    seg_pred = seg_pred.reshape(batch_size, num_seq, num_points, -1).cpu()
    #seg_pred = seg_pred.permute(0, 2, 1, 3).contiguous()
    #seg_pred = seg_pred.reshape(batch_size, num_points, -1).cpu()
    # -----------------------------------------------------------------Point Bert-------------------------------------------------

    # -----------------------------------------------------------------Nearest Neighbor------------------------------------------

    # align sequential frames
    point_clouds = align(point_clouds)

    # Instantiate KNN module
    #knn_module = KNN(k=1, transpose_mode=True)

    # Initialize the result tensor
    # result should have (3xnum_classes) + 11 features for each point
    result = torch.zeros(batch_size, num_points, 68)

    # get num gpus
    num_gpus = faiss.get_num_gpus()
    #print('faiss using num_gpus: ' + str(num_gpus))

    for b in range(batch_size):
        batch_features = []
        pc_t = points_clouds[b, 0]

        for i in range(1, num_seq):
            pc_prev = points_clouds[b, i]

            # Find nearest neighbors in pc_prev using faiss
            # 3 dimensions
            cpu_index = faiss.IndexFlatL2(3)
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

            # add previous point cloud to index
            #print('pc_prev shape: ' + str(pc_prev.shape))
            #print('pc_t shape: ' + str(pc_t.shape))

            gpu_index.add(pc_prev[:,:3].contiguous().numpy())

            k=1
            dist,nearest_neighbor_idx = gpu_index.search(pc_t[:,:3].contiguous().numpy(),k)
            nearest_neighbor_idx = nearest_neighbor_idx.squeeze(-1)
            #print('nearest neighbor idx shape: ' + str(nearest_neighbor_idx.shape))
            #print('nearest neighbor dist shape: ' + str(dist.shape))

            #_, nearest_neighbor_idx = knn_module(pc_prev[:, :3].unsqueeze(0), pc_t[:, :3].unsqueeze(0))

            # Get nearest neighbors from pc_prev using the indices
            nearest_neighbors = pc_prev[nearest_neighbor_idx]
            #print('nearest neighbors shape: ' + str(nearest_neighbors.shape))
            # computing delta p from paper
            #(x,y,z,0) for frame t
            raw_points =torch.cat((pc_t[:,:3],torch.zeros(num_points).unsqueeze(-1)),dim=-1)
            #print('raw points shape: ' + str(raw_points.shape))
            nearest_neighbors = nearest_neighbors - raw_points
            #print('delta p shape: ' + str(nearest_neighbors.shape))
            # concatenating distances as mentioned in paper
            nearest_neighbors = torch.cat((nearest_neighbors,torch.Tensor(dist)),dim=-1)
            # concatenating probability scores
            seg_pred_t = seg_pred[b,i,nearest_neighbor_idx,:]
            #print('seg_pred_t shape: ' + str(seg_pred_t.shape))
            nearest_neighbors = torch.cat((nearest_neighbors,seg_pred_t),dim=-1)
            #print('single time feature shape: ' + str(nearest_neighbors.shape))

            batch_features.append(nearest_neighbors)

        #result[b, :, :4] = points_clouds[b, 0]
        # adding current timestep features to the batch
        current_timestep_reflectance = pc_t[:,-1].unsqueeze(-1)
        #print('current_timestep_reflectances shape: ' + str(current_timestep_reflectance.shape))
        current_timestep_pred = seg_pred[b,0,:,:]
        #print('current_timestep_pred shape: ' + str(current_timestep_pred.shape))
        batch_features.append(torch.concat((current_timestep_reflectance,current_timestep_pred),dim=-1))
        # concatenating all features together for every point
        batch_features = torch.concat(batch_features,dim=-1)
        #print('features for entire batch: ' + str(batch_features.shape))

        # adding this batch feature to the result
        result[b] = batch_features

    del pc_t, pc_prev
    # -----------------------------------------------------------------Nearest Neighbor------------------------------------------


    # concat the prediction and the nearst neighbor info
    #input_seq = torch.cat((seg_pred, result), dim=-1) # ( batch_size, num_points, 4*(num_seq + class_num) )
    input_seq = result

    return {
        'input_seq': input_seq,
        'labels': labels
    }

# version of the dataset after saving predictions (to speed up training)
class SavedP2NetTraining(data.Dataset):
    # saved_preds_path is the path to the root of saved predictions containing all the scenes and frames
    def __init__(self, saved_preds_path, num_points=50000):
        self.num_points = num_points
        # we want to grab the list of every frame which has 2 previous scenes
        # globbing for the 1.pt files
        file_list = Path.glob(Path(saved_preds_path),'*/*/1.pt')

        # filtering out filepaths with frame # 0000 or 00001 (these will not have 2 previous frames)
        filtered_file_list = []
        for file in file_list:
            frame_num = int(file.parent.name)
            if frame_num == 0 or frame_num==1:
                # dont add these
                pass
            else:
                filtered_file_list.append(file)
        self.files = filtered_file_list

    def __getitem__(self, index):
        # 1) pick a file from the file list
        # 2) get previous 2 frames of preds
        # 3) apply knn and concatenation to get a size 68 vector (3x19 + 11)
        # 4) return the feature vector

        # picking a file from the list
        selected_file = self.files[index]

        # getting 3 frames of data [current, current-1, current-2]
        frame_number = int(selected_file.parent.name)
        sequence_root = selected_file.parent.parent

        frame_prev = frame_number -1
        frame_prev_prev = frame_number-2


        #print('int frame number: ' + str(frame_number))
        #print('int prev frame number: ' + str(frame_prev))
        #print('int prev prev number: ' + str(frame_prev_prev))

        # padding with zeros to get 6 digits
        frame_number = str(frame_number)
        frame_prev = str(frame_prev)
        frame_prev_prev = str(frame_prev_prev)

        while len(frame_number)!=6:
            frame_number = '0' + frame_number
        while len(frame_prev) != 6:
            frame_prev = '0' + frame_prev
        while len(frame_prev_prev) !=6:
            frame_prev_prev = '0' + frame_prev_prev

        #print('str frame number: ' + str(frame_number))
        #print('str prev frame number: ' + str(frame_prev))
        #print('str prev prev number: ' + str(frame_prev_prev))

        frame_prev = sequence_root.joinpath(frame_prev,'1.pt')
        frame_prev_prev= sequence_root.joinpath(frame_prev_prev,'1.pt')

        #print('str prev frame path: ' + str(frame_prev))
        #print('str prev prev path: ' + str(frame_prev_prev))

        # getting pytorch data
        # data object has keys 'points', 'labels' 'pred'
        curr_data = torch.load(selected_file)
        prev_data = torch.load(frame_prev)
        prev_prev_data = torch.load(frame_prev_prev)

        pc = [curr_data['points'].numpy(),prev_data['points'].numpy(),prev_prev_data['points'].numpy()]
        # aligning the points clouds
        pc = align(pc)

        # grabbing predictions
        seg_pred = torch.stack([curr_data['pred'],prev_data['pred'],prev_prev_data['pred']]) # (3,num_points,19)

        # knn and generating features
        features = []
        pc_t = pc[0]

        for i in range(1, 3):
            pc_prev = pc[i]

            # Find nearest neighbors in pc_prev using faiss
            # 3 dimensions
            cpu_index = faiss.IndexFlatL2(3)
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

            # add previous point cloud to index
            # print('pc_prev shape: ' + str(pc_prev.shape))
            # print('pc_t shape: ' + str(pc_t.shape))

            gpu_index.add(pc_prev[:, :3].contiguous().numpy())

            k = 1
            dist, nearest_neighbor_idx = gpu_index.search(pc_t[:, :3].contiguous().numpy(), k)
            nearest_neighbor_idx = nearest_neighbor_idx.squeeze(-1)
            # print('nearest neighbor idx shape: ' + str(nearest_neighbor_idx.shape))
            # print('nearest neighbor dist shape: ' + str(dist.shape))

            # _, nearest_neighbor_idx = knn_module(pc_prev[:, :3].unsqueeze(0), pc_t[:, :3].unsqueeze(0))

            # Get nearest neighbors from pc_prev using the indices
            nearest_neighbors = pc_prev[nearest_neighbor_idx]
            # print('nearest neighbors shape: ' + str(nearest_neighbors.shape))
            # computing delta p from paper
            # (x,y,z,0) for frame t
            raw_points = torch.cat((pc_t[:, :3], torch.zeros(self.num_points).unsqueeze(-1)), dim=-1)
            # print('raw points shape: ' + str(raw_points.shape))
            nearest_neighbors = nearest_neighbors - raw_points
            # print('delta p shape: ' + str(nearest_neighbors.shape))
            # concatenating distances as mentioned in paper
            nearest_neighbors = torch.cat((nearest_neighbors, torch.Tensor(dist)), dim=-1)
            # concatenating probability scores
            seg_pred_t = seg_pred[i, nearest_neighbor_idx, :]
            # print('seg_pred_t shape: ' + str(seg_pred_t.shape))
            nearest_neighbors = torch.cat((nearest_neighbors, seg_pred_t), dim=-1)
            # print('single time feature shape: ' + str(nearest_neighbors.shape))

            features.append(nearest_neighbors)

        # result[b, :, :4] = points_clouds[b, 0]
        # adding current timestep features to the batch
        current_timestep_reflectance = pc_t[:, -1].unsqueeze(-1)
        # print('current_timestep_reflectances shape: ' + str(current_timestep_reflectance.shape))
        current_timestep_pred = seg_pred[0, :, :]
        # print('current_timestep_pred shape: ' + str(current_timestep_pred.shape))
        features.append(torch.concat((current_timestep_reflectance, current_timestep_pred), dim=-1))
        # concatenating all features together for every point
        features = torch.concat(features, dim=-1)
        # print('features : ' + str(features.shape))

        # returning the feature vector for this timestep and the labels
        return features, torch.Tensor(curr_data['labels'],type=torch.float32)

    def __len__(self):
        return len(self.files)

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

    dataset = P2Net_Dataset(npoints=50000)

    collate_fn = functools.partial(P2Net_collatn, model=fake_point_bert)

    loader = data.DataLoader(dataset, batch_size=2, collate_fn= collate_fn)

    for i, item in enumerate(loader):
        print(item)
        print(item['input_seq'].shape)
