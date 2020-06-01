import argparse
from torchreid import data_manager
from torchreid import models

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')

# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10,
                    help="print frequency")
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--is_deterministic',action='store_true',help='whether is deterministic')
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")

parser.add_argument('--tr_id_all', type=int, default=1,
                    help="manual seed")


### parameters related to evaluation
parser.add_argument('--evaluate',action='store_true',help="evaluation only")
parser.add_argument('--eval-step', type=int, default=100,help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=100,help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='')
parser.add_argument('--load_weights', type=str, default='',help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--evaluate_by_text_global',action='store_true')


### parameters related to attributes

parser.add_argument('--reid_htri_attribute',action='store_true',help='whether use triplets loss for reid')
parser.add_argument('--attr_aug',action='store_true',help='whether use triplets loss for reid')
parser.add_argument('--coeff_attrib_htri', default=0.1, type=float,help='weights related to re-ID')



### parameters related to data
parser.add_argument('--exp_dir', type=str, default='')
parser.add_argument('--root', type=str, default='data',
                    help="root path to data directory")
parser.add_argument('--root_data', type=str, default='data',
                    help="root path to data directory")
parser.add_argument('--label_path', type=str, default=None,
                    help="path where labels are located")
parser.add_argument('--test_attribute_path', type=str, default=None,
                    help="attribute path")
parser.add_argument('--attribute_path', type=str, default=None,
                    help="attribute path")
parser.add_argument('--attribute_path_bin', type=str, default=None,
                    help="attribute path")
parser.add_argument('--n_class_attr', type=int, default=300,help="")
parser.add_argument('--num_classes_attributes', type=int, default=0,help="")



### parameters related to rank visualization

parser.add_argument('--root_rank', type=str, default='',
                    help="root path to data directory")

parser.add_argument('--log_to_file',action='store_true')



# Datasets
parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                    choices=data_manager.get_names())

parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 4)")


# 256 128
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 384)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")

parser.add_argument('--attr_loss_type', type=str, default='L1',
                    help="norm type")

parser.add_argument('--split-id', type=int, default=0,
                    help="split index")


parser.add_argument('--random_label', type=int, default=0,
                    help="0,1,2")
parser.add_argument('--is_frame',action='store_true')
parser.add_argument('--attraug_reid',action='store_true')

parser.add_argument('--self_attribute_path', type=str, default=None)

parser.add_argument('--global_learning',action='store_true')
parser.add_argument('--htri_learning',action='store_true')



# CUHK03-specific setting
parser.add_argument('--cuhk03_labeled',action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03',action='store_true',
                    help="whether to use cuhk03-metric (default: False)")

# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=120, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=64, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int,
                    help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.000035, type=float,
                    help="initial learning rate")
parser.add_argument('--lrA', '--learning-rate1', default=0.000035, type=float,
                    help="initial learning rate")
parser.add_argument('--lrD', '--learning-rate2', default=0.000035, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[40,70], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")
parser.add_argument('--alphaDKL', type=float, default=0.3,
                    help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity 4")
parser.add_argument('--is_warmup',action='store_true',help="dafault not warmup")
parser.add_argument('--is_REA',action='store_true',help="dafault not rea")
parser.add_argument('--coeff_loss_attributes_reid', default=1, type=float,
                    help="initial learning rate")




