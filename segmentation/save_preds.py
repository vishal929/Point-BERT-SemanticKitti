# saving predictions of model given by pointbert
# this is to alleviate latency of training P2Net
# we sample 50,000 points from the dataset, so we can do the following:
# for each frame, sample 2-3 times, get predictions and save predictions along with point cloud
# when training p2 net, randomly pick predictions from these 2 to 3 samples

from segmentation.data_utils.SemanticKittiDataset import SemanticKitti
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from Constants.constants import ROOT_DIR
from segmentation.models.PointTransformer import get_model
from segmentation.train_kitti import inplace_relu
import os



def generate_preds():

    # firstly making a new directory to hold the saved predictions if they do not already exist
    saved_pred_dir = os.path.join(ROOT_DIR,'segmentation','Saved_Preds')
    if not os.path.exists(saved_pred_dir):
        os.mkdir(saved_pred_dir)

    # loading our model
    # pytorch optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # obtaining torch dataset for training
    npoints = 50000
    train = SemanticKitti(split='train', npoints=npoints, return_files=True, return_reflectance=True)

    # we have 19 usable classes (class 0 is omitted for training and eval)
    num_classes = len(train.inv_map) - 1
    batch_size = 4

    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=16)

    # obtaining model and playing with group size and number of groups
    '''BELOW ARE THE STANDARD GROUP SIZES'''
    # group_size  = 32
    # num_groups = 128
    '''BELOW ARE MY MODIFICATIONS TO GROUP SIZE AND NUM_GROUPS'''

    group_size = 32
    num_groups = 2048
    from easydict import EasyDict
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

    model = get_model(model_config)
    # using data parallelism for multiple gpu training

    # loading model weights
    ckpt = torch.load(os.path.join(ROOT_DIR, 'segmentation', 'saved_weights', 'train_2_results', 'best_model.pth'))
    state_dict = ckpt['model_state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.apply(inplace_relu)
    del (ckpt)
    del (state_dict)

    print('loaded weights... proceeding to generate predictions...')
    torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        # torch.distributed.init_process_group()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        # model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    # number of epochs is how many samples we generate
    num_epochs = 1
    for epoch_num in range(num_epochs):
        with torch.no_grad():
            model = model.eval()

            for batch_id, (points, label, point_files, label_files) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                label = label.long().cuda()
                # we remove remission for running pointbert
                pred_points = points[:,:,:3].float().cuda()

                label = label.long().cuda()

                pred_points = pred_points.transpose(2, 1)
                seg_pred, _ = model(pred_points, None)

                # saving predictions along with points in their own sequence folder
                # i.e
                # .../ROOT_DIR/segmentation/Saved_Preds/Sequence#/frame#/sample#.pty

                for i in range(cur_batch_size):
                   save_obj = {
                       'points': points[i].clone(),
                       'labels': label[i].clone(),
                       'pred':seg_pred[i].clone(),
                       #'point_file':point_files[i], dont need to save this
                       #'label_file':label_files[i], dont need to save this
                   }
                   # getting sequence number
                   sequence = point_files[i]
                   for _ in range(2):
                       sequence = os.path.split(sequence)[0]
                   sequence = os.path.basename(sequence)
                   # getting the frame number  without the '.bin'
                   frame_number = os.path.basename(os.path.splitext(point_files[i])[0])

                   save_dir = os.path.join(saved_pred_dir,sequence)

                   # check if this dir exists
                   if not os.path.exists(save_dir):
                       os.mkdir(save_dir)

                   save_dir = os.path.join(save_dir,frame_number)

                   # check if this dir exists
                   if not os.path.exists(save_dir):
                       os.mkdir(save_dir)

                   # save to .pt torch file
                   torch.save(save_obj,os.path.join(save_dir,str(epoch_num+1)+'.pt'))


