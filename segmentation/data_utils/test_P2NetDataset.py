from segmentation.data_utils.P2NetDataset import P2Net_Dataset, P2Net_collatn
import numpy as np
import torch
import torch.utils.data as data
from easydict import EasyDict
from segmentation.models.PointTransformer import get_model
import functools

# 改dataset， point 的 predict 也要对上

if __name__ == '__main__':
    npoints = 50000
    num_classes = 19
    group_size = 32
    num_groups = 2048
    batch_size = 2
    num_seq = 3

    # get model
    model_config = EasyDict(
        trans_dim=384,
        depth=12,
        drop_path_rate=0.1,
        cls_dim=num_classes,
        num_heads=6,
        group_size=group_size,
        num_group=num_groups,
        encoder_dims=256,
    )
    model = get_model(model_config).to('cuda').requires_grad_(False)

    dataset = P2Net_Dataset(npoints=npoints, num_seq=num_seq)

    collate_fn = functools.partial(P2Net_collatn, model=model)

    loader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    for i, item in enumerate(loader):
        input_seq = item['input_seq']
        labels = item['labels']
        assert input_seq.shape == torch.Size([2, npoints, 4 * (4 + num_classes)])
        break

    # test if the knn works

    result = torch.zeros(batch_size, npoints, 4 * num_seq)

    for i in range(batch_size):
        seq, _ = dataset[i]
        pc_t = torch.tensor(seq[0]).float()
        for j in range(1, num_seq):
            pc_t_1 = torch.tensor(seq[i]).float()
            distance_matrix_1 = torch.cdist(pc_t[:, :3], pc_t_1[:, :3])
            nearest_neighbor_idx = torch.argmin(distance_matrix_1, dim=1)
            nearest_neighbor = pc_t_1[nearest_neighbor_idx]
            result[i, :, 4*j : 4*(j+1)] = nearest_neighbor - pc_t
        result[i, :, :4] = pc_t

    diff = torch.abs(input_seq[:, :, :4*num_seq] - result)
    assert torch.max(diff) < 1e-6
