import functools
from easydict import EasyDict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from segmentation.models.PointTransformer import get_model
from segmentation.data_utils.P2NetDataset import P2Net_Dataset, P2Net_collatn
from segmentation.train_kitti import inplace_relu
from segmentation.p2_net import p2_net

from Constants.constants import ROOT_DIR
import os

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)





def testP2():
    # pytorch optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    npoints = 50000
    num_classes = 19
    group_size = 32
    num_groups = 2048
    batch_size = 1
    num_seq = 3
    lr = 0.003
    num_epochs = 40

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

    ckpt = torch.load('segmentation/saved_weights/best_P2_model.pth')
    state_dict = ckpt['model_state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    p2.load_state_dict(state_dict)
    del (ckpt)
    del (state_dict)
    torch.cuda.empty_cache()
    p2.eval()

    # multiple gpu training
    p2 = torch.nn.DataParallel(p2).cuda()
    model = torch.nn.DataParallel(model).cuda()

    val_set = P2Net_Dataset(npoints=npoints,split='test')

    collate_fn = functools.partial(P2Net_collatn, model=model, split='test')

    val_loader = DataLoader(val_set, batch_size=batch_size,collate_fn=collate_fn, shuffle=False)

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
            seg_pred = p2(input_seq)
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

        print('test Accuracy: %f  mIOU: %f' % (
        test_metrics['accuracy'], test_metrics['total_miou']))