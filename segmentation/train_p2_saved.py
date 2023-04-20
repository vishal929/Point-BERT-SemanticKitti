import functools
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np

from segmentation.models.PointTransformer import get_model
from segmentation.data_utils.P2NetDataset import SavedP2NetTraining,P2Net_Dataset, P2Net_collatn
from segmentation.train_kitti import inplace_relu

from Constants.constants import ROOT_DIR
import os

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class p2_net(nn.Module):
    def __init__(self, q):
        super(p2_net, self).__init__()
        self.q = q
        self.layers = nn.Sequential(
            nn.Linear(3 * q + 11, 128),
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


def trainP2Saved():
    # pytorch optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # creating path to save p2 weights
    checkpoints_dir = os.path.join(ROOT_DIR,'segmentation','saved_weights')

    npoints = 50000
    num_classes = 19
    group_size = 32
    num_groups = 2048
    batch_size = 1
    num_seq = 3
    lr = 0.003
    num_epochs = 10

    # can train with larger batch size on saved preds
    train_batch_size = 32

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
    del (ckpt)
    del (state_dict)
    torch.cuda.empty_cache()
    model.eval()

    # multiple gpu training
    p2 = torch.nn.DataParallel(p2).cuda()
    model = torch.nn.DataParallel(model).cuda()

    saved_preds_path = os.path.join(ROOT_DIR,'segmentation','Saved_Preds')
    train_set = SavedP2NetTraining(saved_preds_path)
    val_set = P2Net_Dataset(npoints=npoints,split='val')

    collate_fn = functools.partial(P2Net_collatn, model=model)

    # need to shuffle when training
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,collate_fn=collate_fn, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(p2.parameters(), lr=lr)

    best_acc = 0
    best_class_avg_iou = 0
    best_total_miou = 0

    for epoch in range(num_epochs):
        print('on training epoch: ' + str(epoch))
        running_loss = 0.0
        mean_correct = []
        p2 = p2.train()
        for i, (features, labels) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            optimizer.zero_grad()

            # features is of shape (batch,npoints,68)
            features = features.to(device)
            preds = p2(features).view(batch_size * npoints, num_classes)
            # import pdb; pdb.set_trace()
            loss = criterion(preds, labels.view(-1))
            loss.backward()
            optimizer.step()

            # getting training accuracy
            correct = preds.eq(labels).type(torch.int32).sum().cpu()
            mean_correct.append(correct.item() / (train_batch_size * npoints))
            running_loss += loss.item()
        epoch_loss = running_loss / batch_size
        print('Epoch %d loss: %.3f' % (epoch + 1, epoch_loss))
        # reporting train accuracy
        train_instance_acc = np.mean(mean_correct)
        print('Train accuracy is: %.5f lr = %.6f' % (train_instance_acc,lr))

        # validation loop
        with torch.no_grad():
            test_metrics = {}
            # we calculate IOU for each class using True positive, False Positive, and False Negative

            true_positive = torch.zeros(num_classes)
            false_positive = torch.zeros(num_classes)
            false_negative = torch.zeros(num_classes)
            total_class_seen = torch.zeros(num_classes)

            p2 = p2.eval()
            for i, item in tqdm(enumerate(val_loader),total=len(val_loader),smoothing=0.9):
                input_seq = item['input_seq'].to(device)
                labels = item['labels'].to(device)

                # computing class level accuracies and miou
                # one hot label has shape (batch,npoints,19)
                one_hot_label = F.one_hot(labels, num_classes)
                seg_pred, _ = p2(input_seq)
                # cur_pred_val = seg_pred.cpu().data.numpy()
                # converting (Batch,npoints,19) to (batch,npoints,1) with the predicted class
                cur_pred_val = torch.argmax(seg_pred, dim=-1)
                # flattening predictions to be (total_points,1) i.e points and their predictions
                cur_pred_val = torch.flatten(cur_pred_val, start_dim=0, end_dim=1)
                # flattening label one_hot to be (total_points,19) i.e one_hot_label for each point
                one_hot_label = torch.flatten(one_hot_label, start_dim=0, end_dim=1)

                # miou needs true positives, false positives, and false negatives for each class
                # one_hot_pred has shape (total_points,19) at this stage
                one_hot_pred = F.one_hot(cur_pred_val, num_classes)
                correct_vals = torch.eq(one_hot_pred, one_hot_label).all(dim=-1)
                true_positive += torch.sum(one_hot_label[correct_vals], dim=0).cpu()
                false_negative += torch.sum(one_hot_label[~correct_vals], dim=0).cpu()
                false_positive += torch.sum(one_hot_pred[~correct_vals], dim=0).cpu()

                # we need to filter out 0 labels values (they are not used for training or testing)

                # getting total number correct by class (we will make use of one hot encodings here)
                total_class_seen += torch.sum(one_hot_label, dim=0).cpu()

                # computing total accuracy, class-wise accuracy, class-wise IoU, and total mIoU
            total_accuracy = torch.nan_to_num(torch.sum(true_positive) / torch.sum(total_class_seen))
            class_accuracy = torch.nan_to_num(true_positive / total_class_seen)

            class_iou = torch.nan_to_num(true_positive / (true_positive + false_negative + false_positive))
            total_miou = torch.sum(class_iou) / num_classes

            test_metrics['accuracy'] = total_accuracy
            test_metrics['class_accuracy'] = class_accuracy
            for cls in range(num_classes):
                print('eval mIoU of %s %f' % (cls, class_iou[cls]))
            test_metrics['class_iou'] = class_iou
            test_metrics['total_miou'] = total_miou

        print('Epoch %d test Accuracy: %f  mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['total_miou']))
        if (test_metrics['total_miou'] >= best_total_miou):
            # logger.info('Save model...')
            print('Save model...')
            savepath = os.path.join(checkpoints_dir, 'best_P2_model.pth')
            print('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_acc': test_metrics['class_accuracy'],
                'class_iou': test_metrics['class_iou'],
                'total_miou': test_metrics['total_miou'],
                'model_state_dict': p2.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            print('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['total_miou'] > best_total_miou:
            best_total_miou = test_metrics['total_miou']
        print('Best accuracy is: %.5f' % best_acc)
        print('Best mIOU is: %.5f' % best_total_miou)