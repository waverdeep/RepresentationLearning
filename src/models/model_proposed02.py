import torch
import torch.nn as nn
import torchvision
import src.models.model_baseline as model_baseline


class PreNetwork(model_baseline.Encoder):
    def __init__(self, input_dim, hidden_dim, strides, filter_sizes, paddings):
        super().__init__(input_dim, hidden_dim, strides, filter_sizes, paddings)


class EncoderNetwork(nn.Module):
    def __init__(self):
        super(EncoderNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1)
        )
        self.vgg16 = torchvision.models.vgg16_bn(pretrained=True, ).features

    def forward(self, x):
        x = self.network(x)
        x = self.vgg16(x)
        return x


class ProjectionNetwork(nn.Module):
    def __init__(self, dimension, hidden_size, projection_size):
        super(ProjectionNetwork, self).__init__()
        self.projection_layer = nn.Sequential(
            nn.Linear(dimension, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.projection_layer(x)


class PredictionNetwork(nn.Module):
    def __init__(self, dimension, hidden_size, prediction_size):
        super(PredictionNetwork, self).__init__()
        self.prediction_layer = nn.Sequential(
            nn.Linear(dimension, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, prediction_size),
        )

    def forward(self, x):
        return self.prediction_layer(x)


class WaveBYOL(nn.Module):
    pass


if __name__ == '__main__':
    pre_network = PreNetwork(
        input_dim=1,
        hidden_dim=512,
        filter_sizes=[10, 8, 4, 4, 4],
        strides=[5, 4, 2, 2, 2],
        paddings=[2, 2, 2, 2, 1],
    ).cuda()

    encoder_network = EncoderNetwork().cuda()

    projector = ProjectionNetwork(64, 512, 2048).cuda()
    predictor = PredictionNetwork(2048, 512, 2048).cuda()

    toy_data = torch.rand(8, 1, 20480).cuda()
    out = pre_network(toy_data)
    print(out.size())

    out = out.unsqueeze(1)
    print(out.size())

    out = encoder_network(out)
    print(out.size())

    ut = out.permute(0, 3, 2, 1)
    print(out.size())

    B, T, D, C = out.shape
    out = out.reshape((B, T, C * D))
    print(out.size())

    out = projector(out)
    print(out.size())

    out = predictor(out)
    print(out.size())





