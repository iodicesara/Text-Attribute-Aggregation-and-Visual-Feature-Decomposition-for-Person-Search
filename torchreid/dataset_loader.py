from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import io

import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    # print(img_path)
    if not osp.exists(img_path):
        print(img_path)
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img



def read_labels(label_path,label_type):
    """to write"""
    got_img = False
    labels=0
    if not osp.exists(label_path):
        return int(0)
    while not got_img:

        file = open(label_path,'r')
        labels=file.read(16)
        got_img = True
    
 
    return int(labels[label_type-1])


def read_labels_from_file_name(img_path,pos):
    """to write"""
    img_path_splitted = img_path.split('/')
    file_name = img_path_splitted[-1];
    #print(pos)
    if pos==-1:
        dir_name = img_path_splitted[-3];
        cam=int(dir_name[3])
    else:
        cam = int(file_name[int(pos-1)])
    #print(cam-1)
    return cam-1


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self,dataset,transform=None,arch='resnetAttW2VAttributes'):
        self.dataset = dataset
        self.transform = transform
        self.arch=arch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        out= self.dataset[index]

        img_path, pid, camid, glove=out[0],out[1],out[2],out[3]

        img = read_image(img_path[0])

        for i in range(len(glove)):
            if isinstance(glove[i], list):
                glove[i] = np.sum(glove[i][:], axis=0)

        if self.transform is not None:
            img = self.transform(img)


        if self.arch=='resnetAttW2VAttributes':

            if len(out) > 4:
                attribute_labels = out[4]

                return img, pid, camid, glove,np.asarray(attribute_labels)

        elif self.arch=='resnetAttW2VText':

            if len(out)>4:
                glove_labels=out[4]
                for i in range(len(glove_labels)):
                    if isinstance(glove_labels[i], list):
                        glove_labels[i] = np.sum(glove_labels[i][:], axis=0)
                return img,pid,camid,glove,glove_labels

        return img,pid,camid




