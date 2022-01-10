import torch.nn as nn
import torch.nn.functional as F
import torch


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
    return torch.mean(2 - 2 * (x * y).sum(dim=-1))


def byol_a_criterion(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def l2_normalization(x):
    x = F.normalize(x, dim=-1, p=2)
    return x


def byol_original_criterion(x, y):
    norm_x = F.normalize(x, dim=-1, p=2)
    norm_y = F.normalize(y, dim=-1, p=2)
    return -2 * torch.mean(torch.sum(x * y) / (norm_x * norm_y))