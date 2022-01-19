import copy
import os
import collections
import torch
import torchvision
import torch.nn as nn
import src.losses.criterion as losses
import src.models.model_proposed02 as model_proposed02
from efficientnet_pytorch import EfficientNet
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class PreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, strides, filter_sizes, paddings):
        super(PreNetwork, self).__init__()
        assert(
                len(strides) == len(filter_sizes) == len(paddings)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.encoder = nn.Sequential()
        for index, (stride, filter_size, padding) in enumerate(zip(strides, filter_sizes, paddings)):
            self.encoder.add_module(
                "encoder_layer_{}".format(index),
                nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                              kernel_size=filter_size, stride=stride, padding=padding, dilation=3),
                    nn.ReLU(),
                )
            )
            input_dim = hidden_dim

    def forward(self, x):
        return self.encoder(x)


if __name__ == '__main__':
    model = PreNetwork(
        input_dim=1,
        hidden_dim=512,
        filter_sizes=[10, 8, 4, 4, 4],
        strides=[5, 4, 2, 2, 2],
        paddings=[2, 2, 2, 2, 1],
    ).cuda()

    input_data01 = torch.rand(8, 1, 20480).cuda()
    otuput = model(input_data01)
    print(otuput.size())

