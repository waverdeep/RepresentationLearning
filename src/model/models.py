import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import numpy as np


class CPC(nn.Module):
    def __init__(self, timestamp, sequence_length):
        super(CPC, self).__init__()
        self.timestamp = timestamp
        self.sequence_length = sequence_length
        # We use five convolutional layers with strides [5, 4, 2, 2, 2],
        # filter-sizes [10, 8, 4, 4, 4] and
        # 512 hidden units with ReLU activations
        self.encoder = nn.Sequential(
            collections.OrderedDict(
                [
                    ('encoder_conv01', nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False)),
                    ('encoder_bn01', nn.BatchNorm1d(512)),
                    ('encoder_relu01', nn.ReLU(inplace=True)),

                    ('encoder_conv02', nn.Conv1d(1, 512, kernel_size=8, stride=4, padding=3, bias=False)),
                    ('encoder_bn02', nn.BatchNorm1d(512)),
                    ('encoder_relu02', nn.ReLU(inplace=True)),

                    ('encoder_conv03', nn.Conv1d(1, 512, kernel_size=4, stride=2, padding=3, bias=False)),
                    ('encoder_bn03', nn.BatchNorm1d(512)),
                    ('encoder_relu03', nn.ReLU(inplace=True)),

                    ('encoder_conv04', nn.Conv1d(1, 512, kernel_size=4, stride=2, padding=3, bias=False)),
                    ('encoder_bn04', nn.BatchNorm1d(512)),
                    ('encoder_relu04', nn.ReLU(inplace=True)),

                    ('encoder_conv05', nn.Conv1d(1, 512, kernel_size=4, stride=2, padding=3, bias=False)),
                    ('encoder_bn05', nn.BatchNorm1d(512)),
                    ('encoder_relu05', nn.ReLU(inplace=True)),
                ]
            )
        )
        # We then use a GRU RNN [17] for the autoregressive part of the model, gar with 256 dimensional hidden state.
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)

        # The output of the GRU at every timestep is used as the context c from which we predict 12 timesteps
        # in the future using the contrastive loss.
        self.Wk = nn.ModuleList([nn.Linear(256, 512) for i in range(self.timestamp)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        # weight initialize
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def forward(self, x, hidden):
        batch = x.size()[0]
        # randomly pick time stamps
        t_sample = torch.randint(self.sequence_length/160-self.timestamp, size=(1, )).long()
        # input sequence: batch*channel*length, x*1*a
        z = self.encoder(x)
        # output sequence: batch*channel*length, x*512*b
        # reshape to batch*length*channel for GRU: x*b*512
        z = z.transpose(1, 2)

        # average over timestamp and batch
        nce = 0
        # encode_sample = torch.empty(self.)



