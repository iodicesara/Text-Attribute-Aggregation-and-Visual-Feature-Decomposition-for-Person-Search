from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.nn.init as init


class ResNet50AttW2VAttribute(nn.Module):

    def __initialize_fc(self, n_in, n_out):
        fc = nn.Linear(n_in, n_out)
        init.normal(fc.weight, std=0.001)
        init.constant(fc.bias, 0)
        return fc

    def __init__(self, num_classes, loss={'xent'}, num_group_attributes=5, num_classes_attributes=(6,4,11,14,6),
                 glove=False,**kwargs):

        super(ResNet50AttW2VAttribute, self).__init__()
        self.loss = loss

        # Backbone
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.num_group_attributes = num_group_attributes

        # re-ID classifier
        self.classifier_reid = nn.Linear(2048, num_classes)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        # Size of features attributes
        self.step = int(2048 / num_group_attributes)

        # Dataset attributes
        self.classifier_list_attributes = nn.ModuleList()

        # Initialize classifier list attributes
        for i in range(num_group_attributes - 1):
            fc = self.__initialize_fc(self.step, num_classes_attributes[i])
            self.classifier_list_attributes.append(fc)

        # Last group
        n_in = int(2048 / num_group_attributes) + ((2048 % num_group_attributes))
        n_out = num_classes_attributes[num_group_attributes - 1]

        # Initialize last fully connected layer
        fc = self.__initialize_fc(n_in, n_out)
        self.classifier_list_attributes.append(fc)

        # Global features for image
        self.ric_features = nn.Linear(2048, 100)

        # Global features for text
        self.fc_text = nn.Linear(300, 100)
        self.fc_text_1 = nn.Linear(100, 100)

        self.training = False
        self.glove = glove

        self.Sigmoid=nn.Sigmoid()
        self.Tahn = nn.Tanh()


    def forward(self, x=None, text=None, only_c=False):

        if x is not None:

            x = self.base(x)
            x = F.avg_pool2d(x, x.size()[2:])
            f = x.view(x.size(0), -1)  # Image features

            logits_list = []

            # attributes prediction
            for i in range(self.num_group_attributes - 1):
                local_feat = f[:, i * self.step: (i + 1) * self.step]
                logits_list.append(self.Sigmoid(self.classifier_list_attributes[i](local_feat)))

            # last attribute prediction
            i = self.num_group_attributes - 1
            local_feat = f[:, i * self.step::]
            logits_list.append(self.Sigmoid(self.classifier_list_attributes[i](local_feat)))


            # global image features
            c_features = self.ric_features(self.Tahn(f))

        if text is not None:
            # global text features
            text_cat = torch.cat(text, dim=1)
            text_features = self.fc_text_1(self.Tahn(self.fc_text(text_cat)))

        if not self.training:
            if x is not None and text is not None:
                return text_features, c_features

            if x is not None:
                return c_features,logits_list

            elif text is not None:
                return text_features

        return logits_list, text_features, c_features



class ResNet50AttW2VText(nn.Module):

    def __initialize_fc(self,n_in,n_out):
        fc = nn.Linear(n_in, n_out)
        init.normal(fc.weight, std=0.001)
        init.constant(fc.bias, 0)
        return fc

    def __init__(self, num_classes, loss={'xent'},num_group_attributes=5,num_features=50,glove=False,**kwargs):
        super(ResNet50AttW2VText, self).__init__()

        num_classes_attributes = [num_features]*num_group_attributes
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.num_group_attributes=num_group_attributes


        # re-ID classifier
        self.classifier_reid = nn.Linear(2048, num_classes)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])


        self.step = int(2048 / num_group_attributes)

        # dataset attributes
        self.classifier_list_attributes = nn.ModuleList()
        self.classifier_list_attributes_1 = nn.ModuleList()


        # initialize classifier list attributes
        for i in range(num_group_attributes-1):

            fc=self.__initialize_fc(self.step,num_classes_attributes[i])
            self.classifier_list_attributes.append(fc)

            fc1 = self.__initialize_fc(num_classes_attributes[i], num_classes_attributes[i])
            self.classifier_list_attributes_1.append(fc1)

        n_in = int(2048 / num_group_attributes) + ((2048 % num_group_attributes))
        n_out = num_classes_attributes[num_group_attributes - 1]

        fc = self.__initialize_fc(n_in, n_out)
        self.classifier_list_attributes.append(fc)

        fc1 = self.__initialize_fc(50, 50)
        self.classifier_list_attributes_1.append(fc1)


        # initialize classifier list attributes7
        self.ric_features = nn.Linear(2048, 100)
        self.fc_text = nn.Linear(50*num_group_attributes, 100)
        self.fc_text_1 = nn.Linear(100, 100)

        self.Tahn = nn.Tanh()
        self.training=False
        self.glove=glove


    def forward(self, x=None,text=None,only_c=False):
        if x is not None:
            x = self.base(x)
            x = F.avg_pool2d(x, x.size()[2:])
            f = x.view(x.size(0), -1)

            logits_list = []

            for i in range(self.num_group_attributes - 1):
                local_feat = f[:, i * self.step: (i + 1) * self.step]
                logits_list.append(self.classifier_list_attributes[i](local_feat))

            i = self.num_group_attributes - 1
            local_feat = f[:, i * self.step::]
            logits_list.append(self.classifier_list_attributes[i](local_feat))

            c_features = self.ric_features(self.Tahn(f))


        if text is not None:
            text_cat = torch.cat(text, dim=1)
            text_features = self.fc_text_1(self.Tahn(self.fc_text(text_cat)))

        if not self.training:
            if x is not None and text is not None:
                return f,text_features,c_features

            if x is not None:
                return c_features,logits_list

            elif text is not None:
                return text_features

        return logits_list,text_features,c_features
