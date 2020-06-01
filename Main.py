from __future__ import print_function
from __future__ import division
import sys
import time
import datetime
import os.path as osp
import os
import random
import numpy as np
import torch
import torch.nn as nn

from torchreid.utils.utils import to_scalar,adjust_lr
from torchreid import models
from torchreid.losses import TripletLoss, LossW2V, TripletLoss1
from torchreid.utils.iotools import save_checkpoint
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.utils.torchtools import count_num_param
from torchreid.optimizers import init_optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from inizialize import initialize_loader
from inizialize import load_weights
from inizialize import resume
from inizialize import test_and_evaluate_dist_mat
from parameters import parser
args = parser.parse_args()

def main():
################################# Setting gpu and out file ###################################################################
    if args.use_cpu: use_gpu = False
    use_gpu=1
    if use_gpu==1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        if args.is_deterministic == 1:
            print("Currently using GPU {}".format(args.gpu_devices))
            print('is deterministic')
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            use_gpu = torch.cuda.is_available()
    else:
        print("Currently using CPU (GPU is highly recommended)")

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))

    print("==========\nArgs:{}\n==========".format(args))
    print("Initializing dataset {}".format(args.dataset))

    [dataset, trainloader_reid, queryloader, galleryloader] = initialize_loader(use_gpu)

    ################################### Initialize model ###############################################################
    if args.num_classes_attributes==0:
        num_classes_attributes = (6,4,11,14,6)

    if args.num_classes_attributes>1:
        num_classes_attributes = (33,6,58,65,9)


    if args.dataset=='pa100K':
        num_classes_attributes = (28, 9, 3, 14, 24, 9)
        if args.num_classes_attributes == 0:
            num_classes_attributes = (5, 3, 3, 6, 9,10)


    if args.dataset=='dukemtmcreid':
        num_classes_attributes = (15, 2,47,62,9)

        if args.num_classes_attributes == 0:
            num_classes_attributes = (2, 2, 8, 11, 6)



    num_group_attributes=len(num_classes_attributes)





    print("Initializing model: {}".format(args.arch))
    if args.arch == 'resnetAttW2VText':
        dim_features = 50*len(num_classes_attributes)

        model = models.init_model(args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'},
                                  num_group_attributes=num_group_attributes,
                                  dim_features=50, glove=True)
        criterion_attributes = LossW2V(num_classes_attributes=num_classes_attributes, attr_loss_type=args.attr_loss_type)

    if args.arch == 'resnetAttW2VAttributes':
        dim_features = sum(num_classes_attributes)

        model = models.init_model(args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'},
                                  num_group_attributes=num_group_attributes,num_classes_attributes=num_classes_attributes,
                                  dim_features=dim_features, glove=True)

        criterion_attributes=LossW2V(num_classes_attributes=num_classes_attributes)


    all_parameters = model.parameters()
    optimizer= init_optim(args.optim, all_parameters, args.lr, args.weight_decay)

    ################################### Loss functions ##############################
    print("Model size: {:.3f} M".format(count_num_param(model)))
    criterion_htri_reid = TripletLoss(margin=0.3); criterion_xent_reid = nn.CrossEntropyLoss();
   

    ################################### Pretrained models ##############################
    if args.load_weights:
       load_weights(model)
    if args.resume:
       resume(model)
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    ################################### Only evaluation ###############################
    writer = SummaryWriter(log_dir=osp.join(args.exp_dir, 'tensorboard'))

    if args.evaluate:
        print("Evaluate by test")
        test_and_evaluate_dist_mat(writer, model, queryloader, galleryloader, use_gpu=use_gpu, save_features=True,
                                   load_features=False,arch=args.arch,size=dim_features)
        return



    ################################### Training ######################################
    start_time = time.time();train_time = 0
    best_rank1 = -np.inf;best_epoch = 0

    # Schedule learning rate
    print("==> Start training");

    for epoch in range(args.start_epoch, args.max_epoch):

        start_train_time = time.time()

        if args.is_warmup:
            adjust_lr(epoch, optimizer, args.lr)

        if args.arch=='resnetAttW2VText':
            train_w2v_single_batch_text(writer, epoch, model, criterion_htri_reid, criterion_attributes, optimizer,
                                        trainloader_reid,use_gpu)

        if args.arch == 'resnetAttW2VAttributes':
            train_w2v_single_batch_attributes(writer, epoch, model, criterion_htri_reid, criterion_attributes,
                                              optimizer, trainloader_reid, use_gpu)

            train_time += round(time.time() - start_train_time)

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                epoch + 1) == args.max_epoch:
            print("==> Test")

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


    state_dict = model.module.state_dict()

    save_checkpoint(state={
         'state_dict': state_dict,
         'epoch': epoch,
    }, fpath=osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))


    print("Evaluate by test")
    test_and_evaluate_dist_mat(writer, model, queryloader, galleryloader, use_gpu=use_gpu, save_features=True,
                                   load_features=False, arch=args.arch, size=dim_features)


def train_w2v_single_batch_attributes(writer, epoch, model, criterion_htri_reid,criterion_attributes, optimizer_reid,
                                      trainloader_reid, use_gpu):

    losses = AverageMeter()

    if args.attraug_reid:
        losses_attributes_reid = AverageMeter()

    if args.global_learning:
        losses_htri_glob_feat = AverageMeter()

    model.train()

    for batch_idx, (imgs, pids, camID,input_glove,output_attributes) in enumerate(trainloader_reid):

        for i in range(len(input_glove)):
            input_glove[i] = input_glove[i].cuda().float()
            output_attributes= output_attributes.cuda().float()

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        logit_attributes, glob_glove, glob_img = model(imgs.squeeze(),input_glove)

        logit_attributes = torch.cat(logit_attributes, dim=1)
        if use_gpu:
            for i in range(len(logit_attributes)):
                logit_attributes[i] = logit_attributes[i].cuda().float()


        glob_feat = torch.cat((glob_glove, glob_img), dim=0)
        pids_tot = torch.cat((pids, pids), dim=0)


        re_loss = 0

        if args.global_learning:
            htri_glob_feat = criterion_htri_reid(glob_feat, pids_tot)
            losses_htri_glob_feat.update(to_scalar(htri_glob_feat), pids.size(0))
            re_loss += htri_glob_feat


        if args.attraug_reid and use_gpu:
            loss_attribute_reid = criterion_attributes(logit_attributes,output_attributes) * args.coeff_loss_attributes_reid
            re_loss += loss_attribute_reid * args.coeff_loss_attributes_reid
            losses_attributes_reid.update(to_scalar(loss_attribute_reid), pids.size(0))

        optimizer_reid.zero_grad()
        losses.update(to_scalar(re_loss), pids.size(0))
        re_loss.backward()
        optimizer_reid.step()

        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss reid and att {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader_reid), loss=losses))

            if args.global_learning:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss cross glob {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch + 1, batch_idx + 1, len(trainloader_reid), loss=losses_htri_glob_feat))

            if args.attraug_reid:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss attributes {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch + 1, batch_idx + 1, len(trainloader_reid), loss=losses_attributes_reid))

    if args.log_to_file:
        writer.add_scalars('train', dict(loss_reid=losses.avg), epoch)

        if args.global_learning:
            writer.add_scalars('train', dict(glob_feat=losses_htri_glob_feat.avg), epoch)

        if args.attraug_reid:
            writer.add_scalars('train', dict(loss_attributes=losses_attributes_reid.avg), epoch)


def train_w2v_single_batch_text(writer, epoch, model, criterion_htri_reid, criterion_attributes,optimizer_reid,
                                trainloader_reid,use_gpu):


    losses = AverageMeter()

    if args.attraug_reid:
        losses_attributes_reid = AverageMeter()
    if args.global_learning:
        losses_htri_glob_feat = AverageMeter()

    model.train()
    if args.htri_learning:
        losses_htri_glob_feat = AverageMeter()
        criterion_htri_reid1 = TripletLoss1(margin=0.3);

    for batch_idx, (imgs, pids, camID,glove,glove_labels) in enumerate(trainloader_reid):

        for i in range(6):
            glove[i] = glove[i].cuda().float()
            glove_labels[i]=glove_labels[i].cuda().float()

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        emb,glob_glove,glob_img= model(imgs.squeeze(),glove)

        glob_feat=torch.cat((glob_glove,glob_img),dim=0)
        pids_tot=torch.cat((pids,pids),dim=0)

        re_loss = 0

        if args.global_learning:
            # cross modality
            htri_glob_feat = criterion_htri_reid(glob_feat, pids_tot)
            losses_htri_glob_feat.update(to_scalar(htri_glob_feat), pids.size(0))
            re_loss += htri_glob_feat
        
        if args.htri_learning:
            htri_glob_feat = criterion_htri_reid1(glob_glove, pids)+criterion_htri_reid1(glob_img, pids)
            losses_htri_glob_feat.update(to_scalar(htri_glob_feat), pids.size(0))
            re_loss += htri_glob_feat



        if args.attraug_reid:
            loss_attribute_reid = criterion_attributes(emb,glove_labels)
            re_loss += loss_attribute_reid*args.coeff_loss_attributes_reid
            losses_attributes_reid.update(to_scalar(loss_attribute_reid), pids.size(0))



        optimizer_reid.zero_grad()
        losses.update(to_scalar(re_loss), pids.size(0))
        re_loss.backward()
        optimizer_reid.step()

        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
            'Loss reid and att {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch + 1,  batch_idx + 1, len(trainloader_reid), loss=losses))

            if args.global_learning or args.htri_learning:
                print('Epoch: [{0}][{1}/{2}]\t'
                'Loss cross glob {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1,  batch_idx + 1, len(trainloader_reid), loss=losses_htri_glob_feat))

            if args.attraug_reid:
                print('Epoch: [{0}][{1}/{2}]\t'
                'Loss attributes {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader_reid), loss=losses_attributes_reid))


    if args.log_to_file:
        writer.add_scalars('train', dict(loss_reid=losses.avg), epoch)
       
        if args.global_learning:
            writer.add_scalars('train', dict(glob_feat=losses_htri_glob_feat.avg), epoch)

        if args.attraug_reid:
            writer.add_scalars('train', dict(loss_attributes=losses_attributes_reid.avg), epoch)



if __name__ == '__main__':
    main()
#

