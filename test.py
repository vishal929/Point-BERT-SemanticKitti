import numpy as np
import torch
def find_nearest_neighbors(pc_t, pc_t_1, pc_t_2):
    # Convert numpy arrays to torch tensors
    pc_t_xyz = torch.tensor(pc_t[:, :3]).float()  # Only use x, y, z coordinates for distance calculations
    pc_t_1_xyz = torch.tensor(pc_t_1[:, :3]).float()
    pc_t_2_xyz = torch.tensor(pc_t_2[:, :3]).float()

    # Calculate the distance matrix
    distance_matrix_1 = torch.cdist(pc_t_xyz, pc_t_1_xyz)
    distance_matrix_2 = torch.cdist(pc_t_xyz, pc_t_2_xyz)

    # Find the indices of the nearest neighbors in pc_t-1 and pc_t-2
    nearest_neighbor_idx_1 = torch.argmin(distance_matrix_1, dim=1)
    nearest_neighbor_idx_2 = torch.argmin(distance_matrix_2, dim=1)

    # Get nearest neighbors from pc_t-1 and pc_t-2 using the indices
    nearest_neighbors_1 = pc_t_1[nearest_neighbor_idx_1.numpy()]
    nearest_neighbors_2 = pc_t_2[nearest_neighbor_idx_2.numpy()]

    # Concatenate pc_t with nearest neighbors from pc_t-1 and pc_t-2
    result = np.hstack((pc_t, nearest_neighbors_1, nearest_neighbors_2))

    return result

"""# Test the function
num_points = 100

# Generate random point clouds
pc_t = np.random.rand(num_points, 4)
pc_t_1 = np.random.rand(num_points, 4)
pc_t_2 = np.random.rand(num_points, 4)

result = find_nearest_neighbors(pc_t, pc_t_1, pc_t_2)
print(result.shape)  # Should output (num_points, 4 * 3)"""

print(torch.cuda.is_available())