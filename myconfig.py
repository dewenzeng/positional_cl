import argparse
import os

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--multiple_device_id", type=tuple, default=(0,1))
parser.add_argument("--num_works", type=int, default=8)
parser.add_argument("--exp_load", type=str, default=None)
parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='results dir')
parser.add_argument('--runs_dir', default='./runs', help='runs dir')

# Data
parser.add_argument("--dataset", type=str, default="chd", help='can be chd, acdc, mmwhs, hvsmr')
parser.add_argument("--data_dir", type=str, default="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/preprocessed_data/2D/")
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument("--enable_few_data", default=False, action='store_true')
parser.add_argument('--sampling_k', type=int, default=10)
parser.add_argument('--cross_vali_num', type=int, default=5)

# Model
parser.add_argument("--initial_filter_size", type=int, default=48)
parser.add_argument("--patch_size", nargs='+', type=int)
parser.add_argument("--classes", type=int, default=4)

# Train
parser.add_argument("--experiment_name", type=str, default="contrast_chd_simclr_")
parser.add_argument("--restart", default=False, action='store_true')
parser.add_argument("--pretrained_model_path", type=str, default='/afs/crc.nd.edu/user/d/dzeng2/UnsupervisedSegmentation/results/supervised_v3_train_2020-10-26_18-41-29/model/latest.pth')
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--decay", type=str, default='50-100-150-200')
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--optimizer", type=str, default='rmsprop',
                    choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
parser.add_argument("--epsilon", type=float, default=1e-8)
parser.add_argument("--do_contrast", default=False, action='store_true')
parser.add_argument("--lr_scheduler", type=str, default='cos')
parser.add_argument("--contrastive_method", type=str, default='simclr', help='simclr, gcl(global contrastive learning), pcl(positional contrastive learning)')

# Loss
parser.add_argument("--temp", type=float, default=0.1)
parser.add_argument("--theta", type=float, default=0.05)
parser.add_argument("--slice_threshold", type=float, default=0.05)

def save_args(obj, defaults, kwargs):
    for k,v in defaults.iteritems():
        if k in kwargs: v = kwargs[k]
        setattr(obj, k, v)

def get_config():
    config = parser.parse_args()
    config.data_dir = os.path.expanduser(config.data_dir)
    config.patch_size = tuple(config.patch_size)
    return config
