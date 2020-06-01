from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import os.path as osp
import shutil
from PIL import Image
import cv2

from .iotools import mkdir_if_missing


def read_labels(label_path,index_file):
    """to write"""
    got_img = False
    labels = 0
    if not osp.exists(label_path):
        raise IOError("{} does not exist".format(label_path))
    while not got_img:
        file = open(label_path, 'r')
        labels = file.read(16)
        got_img = True

    return int(labels[index_file-1])


def _cp_img_to(src, dst, rank, prefix):
    """
    - src: image path or tuple (for vidreid)
    - dst: target directory
    - rank: int, denoting ranked position, starting from 1
    - prefix: string
    """
    if isinstance(src, tuple) or isinstance(src, list):
        dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
        mkdir_if_missing(dst)
        for img_path in src:
            shutil.copy(img_path, dst)
    else:
        dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
        shutil.copy(src, dst)


def read_im(im_path):

    # shape [H, W, 3]
    im = np.asarray(Image.open(im_path))
    # Resize to (im_h, im_w) = (128, 64)
    resize_h_w = (384, 128)
    if (im.shape[0], im.shape[1]) != resize_h_w:
        im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
    # shape [3, H, W]
    im = im.transpose(2, 0, 1)
    return im


def make_im_grid(ims, n_rows, n_cols, space, pad_val):
    """Make a grid of images with space in between.
    Args:
      ims: a list of [3, im_h, im_w] images
      n_rows: num of rows
      n_cols: num of columns
      space: the num of pixels between two images
      pad_val: scalar, or numpy array with shape [3]; the color of the space
    Returns:
      ret_im: a numpy array with shape [3, H, W]
    """
    assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
    assert len(ims) <= n_rows * n_cols
    h, w = ims[0].shape[1:]
    H = h * n_rows + space * (n_rows - 1)
    W = w * n_cols + space * (n_cols - 1)
    if isinstance(pad_val, np.ndarray):
        # reshape to [3, 1, 1]
        pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]
    ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)
    for n, im in enumerate(ims):
        r = n // n_cols
        c = n % n_cols
        h1 = r * (h + space)
        h2 = r * (h + space) + h
        w1 = c * (w + space)
        w2 = c * (w + space) + w
        ret_im[:, h1:h2, w1:w2] = im

    return ret_im


def save_im(im, save_path, i):
    """im: shape [3, H, W]"""
    mkdir_if_missing(save_path)
    im = im.transpose(1, 2, 0)
    Image.fromarray(im).save(save_path + i + '.jpg')



def add_border(im, border_width, value):
    """Add color border around an image. The resulting image size is not changed.
    Args:
      im: numpy array with shape [3, im_h, im_w]
      border_width: scalar, measured in pixel
      value: scalar, or numpy array with shape [3]; the color of the border
    Returns:
      im: numpy array with shape [3, im_h, im_w]
    """
    assert (im.ndim == 3) and (im.shape[0] == 3)
    im = np.copy(im)

    if isinstance(value, np.ndarray):
        # reshape to [3, 1, 1]
        value = value.flatten()[:, np.newaxis, np.newaxis]
    im[:, :border_width, :] = value
    im[:, -border_width:, :] = value
    im[:, :, :border_width] = value
    im[:, :, -border_width:] = value

    return im



def visualize_ranked_results(q_pids, g_pids, q_camids, g_camids, q_img_path,g_img_path,root_rank, root, distmat, dataset, save_dir='log/ranked_results', topk=20):
    # number of query and gallery images
    num_q, num_g = distmat.shape
    print("Visualizing top-{} ranks in '{}' ...".format(topk, save_dir))
    print("# query: {}. # gallery {}".format(num_q, num_g))
    assert num_q == len(dataset.query)
    assert num_g == len(dataset.gallery)

    # indices of gallery images
    indices = np.argsort(distmat, axis=1)

    mkdir_if_missing(save_dir)
    mkdir_if_missing(root_rank + '/all_ranks_' + str(topk))

    count_unmatch=0

    for q_idx in range(num_q):

        qimg_path=q_img_path[q_idx]
        qimg_path=qimg_path[0]
        qpid=q_pids[q_idx]
        qcamid =  q_camids[q_idx]

        qdir = osp.join(save_dir, 'query' + str(q_idx + 1).zfill(5))
        mkdir_if_missing(qdir)

        # _cp_img_to(str(qimg_path), qdir, rank=0, prefix='query')

        ims = [read_im(qimg_path)]

        rank_idx = 1
        not_in_the_first_ranks = 0
        count_g = 0
        miss = False
        g_img_paths=[]
        for g_idx in indices[q_idx, :]:
            gimg_path = g_img_path[g_idx]
            gimg_path = gimg_path[0]
            g_img_paths.append(g_img_path[g_idx])
            gpid = g_pids[g_idx]
            gcamid = g_camids[g_idx]


            invalid = (q_idx==g_idx) and (qcamid == gcamid)

            count_g = count_g + 1
            if not invalid and rank_idx < topk:
                im = read_im(gimg_path)

                # Add green boundary to true positive, red to false positive
                color = np.array([0, 255, 0]) if (qpid == gpid) else np.array([255, 0, 0])
                im = add_border(im, 10, color)

                ims.append(im)
                rank_idx += 1

                if rank_idx==2 and qpid != gpid:
                    miss=True

            if not invalid and rank_idx >= topk and (qpid == gpid):  # blue cases
                im = read_im(gimg_path)
                color = np.array([0, 0, 255])
                im = add_border(im, 10, color)

                ims.append(im)
                rank_idx += 1
                count_unmatch += 1


        im = make_im_grid(ims, 1, len(ims) + 1, 8, 255)

        if miss==True: # match doesn't happen in the first rank positions
            f = open(root_rank + '/all_ranks_' + str(topk) + '/'+ 'desc_' + str(q_idx), "w+")
            f.write(qimg_path)

            for g in g_img_paths:
                f.write(g[0])
            f.close()

            save_im(im, root_rank + '/all_ranks_' + str(topk) + '/', '1_' + str(q_idx))





def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist



