from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLabelSmooth
from .hard_mine_triplet_loss import TripletLoss
from .hard_mine_triplet_loss_1 import TripletLoss1
from .loss_w2v import LossW2V

import torch

def DeepSupervision(criterion, xs, ys):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    # loss = 0.
    # for i in xrange(0,len(xs)):
    #     loss += criterion(xs[i],ys)

    loss = torch.sum(
        torch.stack([criterion(x, ys) for x in xs]))
    loss /= len(xs)
    return loss
