from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import math


class LossW2V(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def cosine_distance(self,x1,x2,eps=1e-8):
        cos=torch.nn.functional.cosine_similarity
        return torch.mean(1.0-cos(x1,x2))


    def RMSELoss(self,yhat,y,eps=1e-6):
        return torch.sqrt(torch.mean((yhat-y)**2)+eps)

    def __init__(self, num_classes_attributes = (8,22,32,32,21,16),weights=None,attr_loss_type='L1'):
        super(LossW2V, self).__init__()
        self.loss=0
        self.num_classes_attributes = num_classes_attributes
        print('L!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if attr_loss_type == 'L1':
            self.loss = nn.L1Loss()
            print('L1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        elif attr_loss_type == 'L2':
            self.loss = nn.MSELoss()
            print('L2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        elif attr_loss_type == 'cos':
            self.loss = self.cosine_distance
            print('cos!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        self.weights=weights



    def remove_zeros(self,label_att,predicted_attributes_group):

        mask=(label_att[:,:]!=0)
        mask=torch.prod(mask,dim=1)
        mask=mask.nonzero().squeeze(-1)
        label_att=label_att[mask,:]
        predicted_attributes_group= predicted_attributes_group[mask,:]
        return label_att,predicted_attributes_group


    def remove_element(self,label_matrix):
        all_mask=[]
        for i in range(0,label_matrix.size()[0]):
            mask=(label_matrix[i,:,:]==0)
            mask=torch.sum(mask,dim=1)
            mask = ((mask!=label_matrix[i,:,:].size()[1]).nonzero())[:,0]
            sel_mask=label_matrix[i,mask,:]
            all_mask.append(sel_mask)
        return all_mask

    def elab_label_attributes(self,label_attributes):
        end_index=0
        start_index=0
        all_label_attributes=[]
        for i in xrange(len(self.num_classes_attributes)):
            start_index = end_index
            end_index = start_index + self.num_classes_attributes[i]
            """ 64x131x50 """
            label_attributes_group = label_attributes[:, start_index:end_index, :]
            label_attributes_group =self.remove_element(label_attributes_group)





    def forward(self, predicted_attributes, label_attributes):
        loss=0
        for i in range(len(label_attributes)):
            loss+=self.loss(predicted_attributes[i],label_attributes[i])

        loss=loss/len(label_attributes)


        return loss



    def _forward(self, predicted_attributes, label_attributes):

        end_index=0
        start_index=0
        loss=0
        for i in xrange(len(self.num_classes_attributes)):
            start_index=end_index
            end_index=start_index+self.num_classes_attributes[i]
            """ 64x131x50 """
            label_attributes_group=label_attributes[:,start_index:end_index,:]
            """ 64x50 """
            predicted_attributes_group=predicted_attributes[i]
            predicted_attributes_group=predicted_attributes_group.unsqueeze(1)
            """ 64x131x50 """
            predicted_attributes_group=predicted_attributes_group.repeat([1,label_attributes_group.size(1),1])
            """ 256x50 """
            predicted_attributes_group=predicted_attributes_group.reshape(predicted_attributes_group.size(0)*predicted_attributes_group.size(1),predicted_attributes_group.size(2))
            """ 256x50 """
            label_attributes_group=label_attributes_group.reshape(label_attributes_group.size(0)*label_attributes_group.size(1),label_attributes_group.size(2))
            """ 64x300 """
            if self.weights!=None:
                W = self.weights[start_index:end_index]
                W = W.unsqueeze(0).unsqueeze(2).expand(label_attributes_group.size(0), -1,
                                                       label_attributes_group.size(2))

                label_att = (label_attributes_group*W).sum(1)
            else:

                label_att,   predicted_attributes_group=self.remove_zeros(label_attributes_group,  predicted_attributes_group)
                # label_att = (label_attributes_group).sum(1)

            part_loss=self.loss(predicted_attributes_group,label_att)

            if torch.isnan(part_loss).any():
                print('error')

            loss=loss+part_loss
        return loss
