import torch.nn as nn
from sklearn.metrics import *


def cosine_similarity(data01, data02):
    cosine = nn.CosineSimilarity(dim=1)
    output = cosine(data01, data02)
    return output.mean()


def pairwise_distance(data01, data02):
    pair = nn.PairwiseDistance(p=2)
    output = pair(data01, data02)
    return output.mean()


def mse(data01, data02):
    output = mean_squared_error(data01, data02)
    return output


def mae(data01, data02):
    output = mean_absolute_error(data01, data02)
    return output