import numpy as np
import torch
from knn_cuda import KNN

# The provided KNN library code should be here

def find_nearest_neighbors_batched(sequences, k=1):
    batch_size, n, num_points, _ = sequences.shape

    # Move sequences to GPU
    sequences = torch.tensor(sequences).float().cuda()

    # Instantiate KNN module
    knn_module = KNN(k=k, transpose_mode=True)

    # Initialize the result tensor
    result = torch.zeros(batch_size, num_points, 4 * n).cuda()

    for b in range(batch_size):
        for i in range(1, n):
            pc_t = sequences[b, i]
            pc_prev = sequences[b, i - 1]

            # Find nearest neighbors in pc_prev
            _, nearest_neighbor_idx = knn_module(pc_prev[:, :3].unsqueeze(0), pc_t[:, :3].unsqueeze(0))

            # Move the nearest neighbor indices back to the CPU
            nearest_neighbor_idx = nearest_neighbor_idx.cpu().numpy().squeeze()

            # Get nearest neighbors from pc_prev using the indices
            nearest_neighbors = pc_prev.cpu().numpy()[nearest_neighbor_idx]

            # Concatenate pc_t with nearest neighbors
            result[b, :, 4 * i:4 * (i + 1)] = torch.tensor(nearest_neighbors).cuda()

        # Set pc_t as the first 4 columns of the result
        result[b, :, :4] = sequences[b, 0]

    # Move the result back to the CPU
    result = result.cpu().numpy()

    return result


# Test the function
B = 2
n = 3
num_points = 100

# Generate random point cloud sequences
sequences = np.random.rand(B, n, num_points, 4)

result = find_nearest_neighbors_batched(sequences)
print(result.shape)  # Should output (B, num_points, 4 * n)