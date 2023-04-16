# test file for semantic kitti semantic segmentation task
from segmentation.data_utils.SemanticKittiDataset import SemanticKitti
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from Constants.constants import ROOT_DIR
from segmentation.models.PointTransformer import get_model
import os


# inplace relu from train_partseg for slight memory savings
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test():
    # pytorch optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # obtaining torch datasets needed for training and setting training parameters
    npoints = 50000
    test = SemanticKitti(split='test',npoints=npoints)

    # we have 19 usable classes (class 0 is omitted for training and eval)
    num_classes = len(test.inv_map) - 1
    batch_size = 4

    test_loader = data.DataLoader(test,batch_size=batch_size,shuffle=False,num_workers=16)

    # obtaining model and playing with group size and number of groups
    '''BELOW ARE THE STANDARD GROUP SIZES'''
    #group_size  = 32
    #num_groups = 128
    '''BELOW ARE MY MODIFICATIONS TO GROUP SIZE AND NUM_GROUPS'''

    group_size=32
    num_groups=2048
    from easydict import EasyDict
    model_config = EasyDict(
        trans_dim= 384,
        depth= 12,
        drop_path_rate= 0.1,
        cls_dim= num_classes,
        num_heads= 6,
        group_size= group_size,
        num_group= num_groups,
        encoder_dims= 256,
    )


    model = get_model(model_config)
    # using data parallelism for multiple gpu training

    # loading model weights
    ckpt = torch.load(os.path.join(ROOT_DIR,'segmentation','saved_weights','train_2_results','best_model.pth'))
    state_dict = ckpt['model_state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.apply(inplace_relu)
    del (ckpt)
    del (state_dict)

    print('loaded weights... proceeding to evaluate...')
    torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        #torch.distributed.init_process_group()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.DataParallel(model)
    model = model.cuda()


    # test loop with mIoU (jaccard index) computation

    with torch.no_grad():
        test_metrics = {}
        # we calculate IOU for each class using True positive, False Positive, and False Negative

        true_positive = torch.zeros(num_classes)
        false_positive = torch.zeros(num_classes)
        false_negative = torch.zeros(num_classes)
        total_class_seen = torch.zeros(num_classes)

        model = model.eval()

        for batch_id, (points, label) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label = points.float().cuda(), label.long().cuda()

            # filtering out 0 class (outlier class not used for training or evaluation)
            # so, we just filter out the points and label for which the label is 0
            # WE ARE FILTERING IN THE DATASET ITSELF SO I COMMENTED THIS OUT
            '''
            remove_zero_mask = ~torch.eq(label, 0)
            points = points[remove_zero_mask]
            label = label[remove_zero_mask]
            '''

            points = points.transpose(2, 1)
            # one hot label has shape (batch,npoints,19)
            one_hot_label = F.one_hot(label,num_classes)
            seg_pred, _ = model(points, None)
            #cur_pred_val = seg_pred.cpu().data.numpy()
            # converting (Batch,npoints,19) to (batch,npoints,1) with the predicted class
            cur_pred_val = torch.argmax(seg_pred,dim=-1)
            # flattening predictions to be (total_points,1) i.e points and their predictions
            cur_pred_val = torch.flatten(cur_pred_val,start_dim=0,end_dim=1)
            # flattening label one_hot to be (total_points,19) i.e one_hot_label for each point
            one_hot_label = torch.flatten(one_hot_label,start_dim=0,end_dim=1)

            #miou needs true positives, false positives, and false negatives for each class
            # one_hot_pred has shape (total_points,19) at this stage
            one_hot_pred = F.one_hot(cur_pred_val,num_classes)
            correct_vals = torch.eq(one_hot_pred,one_hot_label).all(dim=-1)
            true_positive += torch.sum(one_hot_label[correct_vals],dim=0).cpu()
            false_negative += torch.sum(one_hot_label[~correct_vals],dim=0).cpu()
            false_positive += torch.sum(one_hot_pred[~correct_vals],dim=0).cpu()

            # we need to filter out 0 labels values (they are not used for training or testing)

            # getting total number correct by class (we will make use of one hot encodings here)
            total_class_seen += torch.sum(one_hot_label,dim=0).cpu()

        # computing total accuracy, class-wise accuracy, class-wise IoU, and total mIoU
        total_accuracy = torch.nan_to_num(torch.sum(true_positive)/torch.sum(total_class_seen))
        class_accuracy = torch.nan_to_num(true_positive/total_class_seen)

        class_iou = torch.nan_to_num(true_positive/ (true_positive + false_negative + false_positive))
        total_miou = torch.sum(class_iou)/num_classes

        test_metrics['accuracy'] = total_accuracy
        test_metrics['class_accuracy'] = class_accuracy
        for cls in range(num_classes):
            print('test mIoU of %s %f' % (cls, class_iou[cls]))
            print('test acc of %s %f' % (cls,class_accuracy[cls]))
        test_metrics['class_iou'] = class_iou
        test_metrics['total_miou'] = total_miou
        print('test overall accuracy: %f mIoU: %f' % (test_metrics['accuracy'],test_metrics['total_miou']))
