from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class TripletLoss1(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss1, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        mask_modality_1=torch.zeros(int(targets.shape[0]/2),dtype=torch.uint8).cuda()
        mask_modality_2=torch.ones(int(targets.shape[0]/2),dtype=torch.uint8).cuda()

        mask_modality_first_part=torch.cat((mask_modality_1,mask_modality_2)).cuda()
        mask_modality_first_part=mask_modality_first_part.repeat((64,1)).cuda()

        mask_modality_second_part=torch.cat((mask_modality_2,mask_modality_1)).cuda()
        mask_modality_second_part=mask_modality_second_part.repeat((64,1)).cuda()

        mask_modality_final=torch.cat((mask_modality_first_part,mask_modality_second_part),dim=0).cuda()
        mask_modality_final_neg=torch.cat((mask_modality_second_part,mask_modality_first_part),dim=0).cuda()

        mask_positives=mask*mask_modality_final;
        mask_negatives=mask+mask_modality_final_neg;

        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask_positives[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask_negatives[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class TripletLossAttribute(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLossAttribute, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def __compute_dist(self,inputs,n):
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    #
    def __compute_dist_row_wise_1(self, features, weights):
         n = features.size(0)
         dist = torch.zeros((n, n)).cuda()
    #
         for i in range(n):
             for j in range(n):
                 sq_dist = torch.sum(((features[i, :] * weights[i, j, :]) - (features[j, :] * weights[i, j, :])) ** 2, 0)
                 dist[i, j] = sq_dist.clamp(min=1e-12).sqrt()  # for numerical stability

         return dist

    def __compute_dist_row_wise(self,features,weights):

        feat1=features.unsqueeze(dim=0).expand(64,64,2048)
        feat2=features.unsqueeze(dim=1).expand(64,64,2048)
        sq_dist = torch.sum(((feat1*weights)-(feat2*weights)) ** 2,dim=2).clamp(min=1e-12).sqrt()  # for numerical stability

        return sq_dist


    def __intersection_attribute(self,attributes):
        n=attributes.size(0)
        attribute_intersection=torch.zeros((n,n))
        for i in range(attributes.size(0)):
            for j in range(attributes.size(0)):
                tmp=torch.sum(torch.mul(attributes[i,:],attributes[j,:]))/attributes.size(1)
                attribute_intersection[i,j]=tmp

        return attribute_intersection


    def __intersection_attribute_vec_wise(self,attributes,m):
        n=attributes.size(0)

        attribute_intersection=torch.zeros((n,n,m)).cuda()
        for i in range(attributes.size(0)):
            for j in range(attributes.size(0)):
                tmp=(torch.sum(torch.eq(attributes[i,:],attributes[j,:]))).float()/attributes.size(1)
                attribute_intersection[i,j,:]=tmp.expand(m)

        return attribute_intersection

    def __intersection_attribute_vec(self,attributes,m):
        n=attributes.size(1)
        feat1 = attributes.unsqueeze(dim=0).expand(64, 64,n)
        feat2 = attributes.unsqueeze(dim=1).expand(64, 64,n)

        attribute_intersection=(torch.sum(torch.eq(feat1,feat2),dim=2).float()/n).unsqueeze(2).expand(64,64,m)
        return attribute_intersection





    def __get_disentangled_features_attributes(self,inputs,attributes,num_class_attributes):
        disentangled_features = []
        disentangled_attributes = []
        start = 0;
        start_attrib = 0
        offset = int(inputs.size(1) / len(num_class_attributes))
        remaining = inputs.size(1) % len(num_class_attributes)
        for i in range(len(num_class_attributes) - 1):
            end = start + offset
            disentangled_features.append(inputs[:, start:end])
            disentangled_attributes.append(attributes[:, start_attrib:start_attrib + num_class_attributes[i]])
            start = end
            start_attrib = num_class_attributes[i]

        end = start + remaining
        disentangled_features.append(inputs[:, start:end])
        disentangled_attributes.append(attributes[:, start_attrib:start_attrib + num_class_attributes[len(num_class_attributes) - 1]])
        return disentangled_features,disentangled_attributes


    def __sum_distances(self,disentangled_distances, disentangled_intersection_attr, is_positive=True):

        n=disentangled_distances[0].size(0)

        for i in range(len(disentangled_distances)):
            if is_positive==True:
                if i==0:
                    dist=disentangled_distances[i]*disentangled_intersection_attr[i].cuda()
                else:
                    dist=dist+disentangled_distances[i]*disentangled_intersection_attr[i].cuda()
            else:
                if i==0:
                    dist=disentangled_distances[i]*(1-disentangled_intersection_attr[i]).cuda()
                else:
                    dist = dist + disentangled_distances[i] * (1-disentangled_intersection_attr[i]).cuda()



        return dist

    def __sum_distances1(self,disentangled_distances):

        n=disentangled_distances[0].size(0)

        for i in range(len(disentangled_distances)):

            if i==0:
                dist=disentangled_distances[i]
            else:
                dist=dist+disentangled_distances[i]

        return dist




    def forward(self, inputs, targets,attention_attribute):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)


        dimen=(341,341,341,341,341,343)
        attention_attribute=attention_attribute.unsqueeze(dim=2).view(64,64,-1)
        attention_list=[]
        for i in range(len(dimen)):
            attention_list.append(attention_attribute[:,:,i].unsqueeze(dim=2).expand(64,64,dimen[i]))

        disentangled_intersection_attr =torch.cat(attention_list, dim=2)



        dist_pos = self.__compute_dist_row_wise(inputs, disentangled_intersection_attr)
        dist_neg = self.__compute_dist_row_wise(inputs, disentangled_intersection_attr)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []



        for i in range(n):
            dist_ap.append(dist_pos[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist_neg[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
