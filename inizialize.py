from torchreid import transforms as T
from torchreid import data_manager
from parameters import parser
from torchreid.dataset_loader import ImageDataset
from torch.utils.data import DataLoader
from torchreid.samplers import RandomIdentitySampler
from torchreid.eval_metrics import evaluate,eval_market1501_multiple_g
from torchreid.utils.iotools import check_isfile

import time
from torch.autograd import Variable
from torchreid.utils.avgmeter import AverageMeter

import os.path as osp
import random
import numpy as np
import torch

args = parser.parse_args()


def initialize_single_batch():

    if args.dataset=='market1501' or args.dataset=='dukemtmcreid' or args.dataset=='pa100K':
        dataset_reid = data_manager.init_imgreid_dataset(
            root=args.root, name=args.dataset, split_id=args.split_id,
            cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
            attribute_path=args.attribute_path,
            attribute_path_bin=args.attribute_path_bin, random_label=args.random_label, is_frame=args.is_frame,
            self_attribute_path=args.self_attribute_path, arch=args.arch,test_attribute_path=args.test_attribute_path,
            tr_id_all=args.tr_id_all)

    random.shuffle(dataset_reid.train)
    return dataset_reid


def set_transform():

    if args.is_REA:
        transform_train = T.Compose([
            T.Random2DTranslation(args.height, args.width),
            T.RandomHorizontalFlip(),
            T.RandomEraising(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform_train = T.Compose([
            T.Random2DTranslation(args.height, args.width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_train,transform_test


def get_trainloader_resnetAttr(dataset_reid,transform_train,pin_memory):

    if args.dataset=='market1501' or args.dataset=='dukemtmcreid' or args.dataset=='pa100K':
        trainloader_reid = DataLoader(
            ImageDataset(dataset_reid.train,transform=transform_train,arch=args.arch),
            sampler=RandomIdentitySampler(dataset_reid.train, args.train_batch, args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )

    return trainloader_reid




def initialize_loader(use_gpu):

    dataset_reid=initialize_single_batch()

    pin_memory = True if use_gpu else False
    transform_train,transform_test=set_transform()

    train_loader_reid = get_trainloader_resnetAttr(dataset_reid, transform_train, pin_memory)

    queryloader = DataLoader(ImageDataset(dataset_reid.query, transform=transform_test,arch=args.arch),
                             batch_size=args.test_batch, shuffle=False, num_workers=args.workers,pin_memory=pin_memory,
                             drop_last=False,
    )

    galleryloader = DataLoader(ImageDataset(dataset_reid.gallery, transform=transform_test,arch=args.arch),
                               batch_size=args.test_batch, shuffle=False, num_workers=args.workers,pin_memory=pin_memory,
                               drop_last=False,
    )

    return [dataset_reid, train_loader_reid, queryloader, galleryloader]



def load_weights(model):
    # load pretrained weights but ignore layers that don't match in size
    if check_isfile(args.load_weights):
        checkpoint = torch.load(args.load_weights)
        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                         k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))


def resume(model):
    if check_isfile(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        rank1 = checkpoint['rank1']
        print("Loaded checkpoint from '{}'".format(args.resume))
        print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, rank1))



def  compute_global_descriptor_from_text(loader, use_gpu, model,arch,size=0):
    batch_time = AverageMeter()
    model.training=False
    model.glove=True

    qf=np.zeros([len(loader),100],dtype=float)

    qf_glove = np.zeros([len(loader), size], dtype=float)
    q_pids= np.zeros([len(loader)],dtype=float)
    q_camids=np.zeros([len(loader)],dtype=float)

    for batch_idx, out in enumerate(loader):

        pids=out[1]; camids=out[2];

        if arch=='resnetAttW2VAttributes':
            text_desc = out[4]
            attribute_text=out[3]
        elif arch=='resnetAttW2VText':
            attribute_text = out[4]
            text_desc = torch.cat(attribute_text, dim=1)

        print(str(batch_idx) + '/' + str(len(loader)))
        if use_gpu:
            feat= model(text=attribute_text)
            feat=feat.squeeze()

        qf[batch_idx] = feat.cpu()
        qf_glove[batch_idx]=text_desc
        q_pids[batch_idx] = np.asarray(pids)
        q_camids[batch_idx] = np.asarray(camids)

    return qf,qf_glove,q_pids,q_camids



def  compute_global_descriptor_from_image(loader, use_gpu, model,arch,size):
    batch_time = AverageMeter()
    model.training=False
    model.glove=True

    qf=np.zeros([len(loader),100],dtype=float)
    qf_local=np.zeros([len(loader),size],dtype=float)
    q_pids= np.zeros([len(loader)],dtype=float)
    q_camids=np.zeros([len(loader)],dtype=float)
    grove_dic={}


    for batch_idx, out in enumerate(loader):

        imgs=out[0]; pids=out[1]; camids=out[2];

        print(str(batch_idx) + '/' + str(len(loader)))

        if use_gpu:
            imgs = imgs.cuda()

        imgs = Variable(imgs, volatile=True)
        if len(imgs.size())>4:
            b, n, s, c, h, w = imgs.size()
            assert (b == 1)
            imgs = imgs.view(b * n, s, c, h, w)

        # print(imgs.size())
        imgs = imgs.squeeze()

        num_iter = 1
        if imgs.size(0) > 100:
            num_iter = int(np.ceil(float(imgs.size(0)) / 100))
            batch_size = 100
        else:
            num_iter = 1
            if len(imgs.size())>3:
                batch_size = imgs.size(0)
            else:
                batch_size=0
        features = []
        local_features=[]
        for iii in range(num_iter):

            if batch_size>0:
                start_index = iii * batch_size
                end_index = iii * batch_size + batch_size

                if end_index > imgs.size(0):
                    end_index = imgs.size(0)

                batch_size=end_index-start_index

                img = imgs[start_index:end_index, :, :, :]
            else:
                img=imgs.unsqueeze(dim=0)

            feat,local_feat= model(x=img,only_c=True)
            local_feat = torch.cat(local_feat, dim=1)

            if arch=='resnetAttW2VAttributes':
                local_feat=torch.round(local_feat)


            if batch_size>0:
                feat=feat.mean(dim=0)
                feat=feat.unsqueeze(dim=1)
                local_feat = local_feat.mean(dim=0)
                local_feat = local_feat.unsqueeze(dim=1)

            features.append(feat)
            local_features.append(local_feat)

        len_feat=len(features)
        features = torch.cat(features, dim=1)
        local_features=torch.cat(local_features,dim=1)
        if len_feat>1:
            features = features.mean(dim=1)
            local_features= local_features.mean(dim=1)

        qf[batch_idx] = features.squeeze().cpu()
        qf_local[batch_idx] = local_features.squeeze().cpu()

        q_pids[batch_idx] = np.asarray(pids)
        q_camids[batch_idx] = np.asarray(camids)


    return qf,qf_local,q_pids,q_camids



def test_and_evaluate_dist_mat(writer, model, queryloader, galleryloader, use_gpu,save_features=False,
                               load_features=False,arch=None,size=0):


    if load_features==True:
        qf=np.load(osp.join(args.save_dir,'qf_0.npy'))
        qf_0=np.load(osp.join(args.save_dir,'qf_local_0.npy'))
        q_pids=np.load(osp.join(args.save_dir,'q_pids_0.npy'))
        q_camids=np.load(osp.join(args.save_dir,'q_camids_0.npy'))

        gf = np.load(osp.join(args.save_dir, 'gf_0.npy'))
        gf_0 = np.load(osp.join(args.save_dir, 'gf_local_0.npy'))
        g_pids = np.load(osp.join(args.save_dir, 'g_pids_0.npy'))
        g_camids = np.load(osp.join(args.save_dir, 'g_camids_0.npy'))


    else:
        [qf,qf_0, q_pids, q_camids, gf,gf_0, g_pids, g_camids] = test(model,queryloader,galleryloader,use_gpu,arch=arch,
                                                                      size=size)

        if save_features == True:
            np.save(osp.join(args.save_dir, 'qf_0.npy'), qf)
            np.save(osp.join(args.save_dir, 'qf_local_0.npy'), qf_0)
            np.save(osp.join(args.save_dir, 'q_pids_0.npy'), q_pids)
            np.save(osp.join(args.save_dir, 'q_camids_0.npy'), q_camids)
            np.save(osp.join(args.save_dir, 'gf_0.npy'), gf)
            np.save(osp.join(args.save_dir, 'gf_local_0.npy'), gf_0)
            np.save(osp.join(args.save_dir, 'g_pids_0.npy'), g_pids)
            np.save(osp.join(args.save_dir, 'g_camids_0.npy'), g_camids)

    if arch=='resnetAttW2VAttributes':
        evaluation(qf, q_pids, q_camids, gf, g_pids, g_camids, qf_0=qf_0,gf_0=gf_0,hamming=True)
    else:
        evaluation( qf, q_pids, q_camids, gf, g_pids, g_camids,qf_0=qf_0,gf_0=gf_0)






# add the protocol for video https://github.com/jiyanggao/Video-Person-ReID/blob/master/video_loader.py
def test(model, queryloader, galleryloader, use_gpu, arch=None,size=0):

    model.eval()
    with torch.no_grad():
        qf,qf_local, q_pids, q_camids = compute_global_descriptor_from_text(queryloader, use_gpu,model,arch, size)
        gf,gf_local, g_pids, g_camids = compute_global_descriptor_from_image(galleryloader, use_gpu, model,arch,size)
        return [qf, qf_local,q_pids, q_camids, gf,gf_local, g_pids, g_camids]



def print_evaluation(cmc,mAP,ranks):
    #
    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

def re_ranking(indices,re_rank_index,distmat,g_pids,g_camids,g_bias_labels,q_pids,q_camids,q_bias_labels):
    r_distmat = np.zeros((indices.shape[0], re_rank_index))
    r_g_pids = np.zeros((indices.shape[0], re_rank_index))
    r_g_camids = np.zeros((indices.shape[0], re_rank_index))
    r_g_bias_labels = np.zeros((indices.shape[0], re_rank_index))
    for k in range(0, indices.shape[0]):
        r_distmat[k, :] = distmat[k, indices[k, 0:re_rank_index]]
        r_g_pids[k, :] = g_pids[indices[k, 0:re_rank_index]]
        r_g_camids[k, :] = g_camids[indices[k, 0:re_rank_index]]
        r_g_bias_labels[k, :] = g_bias_labels[indices[k, 0:re_rank_index]]

    cmc, mAP = eval_market1501_multiple_g(r_distmat, q_pids, r_g_pids, q_camids, r_g_camids, q_bias_labels,
                                          r_g_bias_labels, max_rank=100,
                                          disable_negative_bias=False, is_bias=True)
    ranks = [1, 5, 10, 20, 30, 50]
    print_evaluation(cmc, mAP,ranks)


def evaluation(qf, q_pids, q_camids, gf, g_pids, g_camids,hamming=False,qf_0=None,gf_0=None):

    m, n = len(qf), len(gf)
    if qf_0 is not None and gf_0 is not None:
        qf1 = torch.from_numpy(qf_0)
        gf1 = torch.from_numpy(gf_0)


    qf0=torch.from_numpy(qf)
    gf0=torch.from_numpy(gf)


    distmat = torch.pow(qf0, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf0, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf0, gf0.t())
    distmat = distmat.numpy()/100

    if qf_0 is not None and gf_0 is not None:

        if hamming == True:
            import scipy.spatial.distance as distt
            distmat1 = distt.cdist(qf1.numpy(), gf1.numpy(), 'hamming')/171
        else:
            distmat1 = torch.pow(qf1, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf1, 2).sum(dim=1,keepdim=True).expand(n, m).t()
            distmat1.addmm_(1, -2, qf1, gf1.t())
            distmat1 = distmat1.numpy()/250


        for i in range(0,11):
            a = 0.1*i
            b = 1-a
            distmat_1 = b * distmat + a * distmat1
            print("Computing CMC and mAP"+str(a)+"::"+str(b))
            cmc, mAP, indices = evaluate(distmat_1, q_pids, g_pids, q_camids, g_camids)
            ranks = [1, 5, 10, 20, 30, 50, 100, 200]
            print_evaluation(cmc,mAP,ranks)

