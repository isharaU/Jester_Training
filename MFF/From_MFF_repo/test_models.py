import argparse
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule
import datasets_video
import torchvision
from torch.nn import functional as F
import logging

# Disable SSL verification (for downloading pretrained models)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Options
parser = argparse.ArgumentParser(description="MFF testing on the full validation set")
parser.add_argument('dataset', type=str, choices=['jester', 'nvgesture', 'chalearn'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff', 'RGBFlow'])
parser.add_argument('weights', type=str, help="Path to the pretrained model weights")
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None, help="Path to save output scores")
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1, help="Maximum number of samples to test")
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--num_motion', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='MLP', choices=['avg', 'MLP'])
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--gpus', nargs='+', type=int, default=None, help="GPU IDs to use")
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--num_set_segments', type=int, default=1)
parser.add_argument('--softmax', type=int, default=0, help="Apply softmax to model output if set to 1")
parser.add_argument('--fp16', action='store_true', help="Enable mixed precision (FP16) inference")

args = parser.parse_args()

# Set dataset paths
args.root_path = "/content/drive/MyDrive/V2E/test/jester/flow"
args.val_list = "/content/drive/MyDrive/V2E/test/jester/jester-v1-validation.csv"

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logger.warning("No GPU available, using CPU instead.")

# Mixed precision support
if args.fp16:
    from torch.cuda.amp import autocast
    logger.info("Mixed precision (FP16) enabled.")


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Load dataset categories
categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality)
num_class = len(categories)

# Initialize model
net = TSN(num_class, args.test_segments if args.consensus_type in ['MLP'] else 1, args.modality,
          base_model=args.arch,
          consensus_type=args.consensus_type,
          img_feature_dim=args.img_feature_dim)

# Load pretrained weights
logger.info(f"Loading model weights from {args.weights}")
checkpoint = torch.load(args.weights, map_location=device)
logger.info(f"Model epoch {checkpoint['epoch']}, best prec@1: {checkpoint['best_prec1']}")

base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
net = net.to(device)
net.eval()

# Data preprocessing
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError(f"Unsupported number of test crops: {args.test_crops}")

# Data loading
data_length = 1 if args.modality == 'RGB' else 5 if args.modality in ['Flow', 'RGBDiff'] else args.num_motion

data_loader = torch.utils.data.DataLoader(
    TSNDataSet(args.root_path, args.val_list, num_segments=args.test_segments,
               new_length=data_length,
               modality=args.modality,
               image_tmpl=prefix,
               dataset=args.dataset,
               test_mode=True,
               transform=torchvision.transforms.Compose([
                   cropping,
                   Stack(roll=(args.arch in ['BNInception', 'InceptionV3']), isRGBFlow=(args.modality == 'RGBFlow')),
                   ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                   GroupNormalize(net.input_mean, net.input_std),
               ])),
    batch_size=1, shuffle=False,
    num_workers=args.workers * 2, pin_memory=False)

# Evaluation function
def eval_video(video_data):
    i, data, label = video_data
    data = data.to(device)
    label = label.to(device)

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    elif args.modality == 'RGBFlow':
        length = 3 + 2 * args.num_motion
    else:
        raise ValueError(f"Unknown modality: {args.modality}")

    with torch.no_grad():
        if args.fp16:
            with autocast():
                input_var = data.view(-1, length, data.size(2), data.size(3))
                rst = net(input_var)
        else:
            input_var = data.view(-1, length, data.size(2), data.size(3))
            rst = net(input_var)

    if args.softmax == 1:
        rst = F.softmax(rst, dim=1)

    rst = rst.cpu().numpy()
    if args.consensus_type in ['MLP']:
        rst = rst.reshape(-1, 1, num_class)
    else:
        rst = rst.reshape((args.test_crops, args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

    return i, rst, label[0]


# Main evaluation loop
proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

top1 = AverageMeter()
top5 = AverageMeter()
output = []

for i, (data, label) in enumerate(data_loader):
    if i >= max_num:
        break

    try:
        rst = eval_video((i, data, label))
        output.append(rst[1:])
        cnt_time = time.time() - proc_start_time
        prec1, prec5 = accuracy(torch.from_numpy(np.mean(rst[1], axis=0)), label, topk=(1, 5))
        top1.update(prec1[0], 1)
        top5.update(prec5[0], 1)
        logger.info(f'Video {i} done, total {i + 1}/{total_num}, average {cnt_time / (i + 1):.3f} sec/video, '
                    f'moving Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
    except Exception as e:
        logger.error(f"Error processing video {i}: {e}")
        continue

# Compute confusion matrix and final metrics
video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
video_labels = [x[1] for x in output]

cf = confusion_matrix(video_labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
cls_acc = cls_hit / cls_cnt

logger.info('-----Evaluation is finished------')
logger.info(f'Class Accuracy {np.mean(cls_acc) * 100:.02f}%')
logger.info(f'Overall Prec@1 {top1.avg:.02f}% Prec@5 {top5.avg:.02f}%')

# Save results if required
if args.save_scores:
    name_list = [x.strip().split()[0] for x in open(args.val_list)]
    order_dict = {e: i for i, e in enumerate(sorted(name_list))}
    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)
    reorder_pred = [None] * len(output)
    output_csv = []

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]
        reorder_pred[idx] = video_pred[i]
        output_csv.append(f'{name_list[i]};{categories[video_pred[i]]}')

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label, predictions=reorder_pred, cf=cf)
    with open(args.save_scores.replace('npz', 'csv'), 'w') as f:
        f.write('\n'.join(output_csv))
    logger.info(f"Results saved to {args.save_scores}")