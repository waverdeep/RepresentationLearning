import torch.nn as nn
import torch.nn.functional as F


def set_criterion(name, params=None):
    if name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'L1Loss':
        return nn.L1Loss()
    elif name == 'NLLLoss':
        return nn.NLLLoss()
    elif name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()


def byol_criterion(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)
