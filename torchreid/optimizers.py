from __future__ import absolute_import

import torch


def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'amsgrad':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optimizer: {}".format(optim))



def find_index(seq, item):
  for i, x in enumerate(seq):
    if item == x:
      return i
  return -1


def adjust_lr_staircase(param_groups, base_lrs, ep, decay_at_epochs, factor):
    """Multiplied by a factor at the BEGINNING of specified epochs. Different
    param groups specify their own base learning rates.

    Args:
      param_groups: a list of params
      base_lrs: starting learning rates, len(base_lrs) = len(param_groups)
      ep: current epoch, ep >= 1
      decay_at_epochs: a list or tuple; learning rates are multiplied by a factor
        at the BEGINNING of these epochs
      factor: a number in range (0, 1)

    Example:
      base_lrs = [0.1, 0.01]
      decay_at_epochs = [51, 101]
      factor = 0.1
      It means the learning rate starts at 0.1 for 1st param group
      (0.01 for 2nd param group) and is multiplied by 0.1 at the
      BEGINNING of the 51'st epoch, and then further multiplied by 0.1 at the
      BEGINNING of the 101'st epoch, then stays unchanged till the end of
      training.

    NOTE:
      It is meant to be called at the BEGINNING of an epoch.
    """
    assert len(base_lrs) == len(param_groups), \
        "You should specify base lr for each param group."
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep not in decay_at_epochs:
        return

    ind = find_index(decay_at_epochs, ep)
    for i, (g, base_lr) in enumerate(zip(param_groups, base_lrs)):
        g['lr'] = base_lr * factor ** (ind + 1)
        print('=====> Param group {}: lr adjusted to {:.10f}'
              .format(i, g['lr']).rstrip('0'))