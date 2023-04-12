import functools
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from segmentation.models.PointTransformer import get_model
from segmentation.data_utils.P2NetDataset import P2Net_Dataset, P2Net_collatn
from segmentation.train_kitti import inplace_relu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class p2_net(nn.Module):
    def __init__(self, q):
        super(p2_net, self).__init__()
        self.q = q
        self.layers = nn.Sequential(
            nn.Linear(3*q+11, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, q),
            nn.Softmax()
        )

    def forward(self, x):
        batch_size, n, _ = x.shape
        x = x.view(batch_size * n, -1)
        x = self.layers(x)
        x = x.view(batch_size, n, self.q)
        
        return x

def trainP2():
    # pytorch optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    npoints = 50000
    num_classes = 19
    group_size = 32
    num_groups = 2048
    batch_size = 1
    num_seq = 3
    lr= 0.003
    num_epochs = 10

    p2 = p2_net(q=num_classes).to(device)
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
    ckpt = torch.load('segmentation/saved_weights/train_2_results/best_model.pth')
    state_dict = ckpt['model_state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.apply(inplace_relu)
    del(ckpt)
    del(state_dict)
    torch.cuda.empty_cache()
    model.eval()

    dataset = P2Net_Dataset(npoints=npoints, num_seq=num_seq)

    collate_fn = functools.partial(P2Net_collatn, model=model)

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, item in enumerate(loader):
            optimizer.zero_grad()
            input_seq = item['input_seq'].to(device)
            labels = item['labels'].to(device)

            outputs = p2(input_seq).view(batch_size*npoints, num_classes)
            # import pdb; pdb.set_trace()
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / batch_size
        print('Epoch %d loss: %.3f' % (epoch+1, epoch_loss))