# thanks for https://github.com/nttcslab/byol-a
import copy

import torch
import torch.nn as nn
import src.losses.criterion as criterion

def set_requires_grad(model, requires):
    for parameter in model.parameters():
        parameter.requires_grad = requires


class BYOL(nn.Module):
    def __init__(self, input_dims, hidden_dims, strides, filter_sizes, paddings,
                 maxpool_filter_sizes, maxpool_strides, feature_dimension,
                 hidden_size, projection_size):
        super(BYOL, self).__init__()
        dimension = feature_dimension
        # setup online network
        self.online_encoder = EncodingNetwork(
            input_dims, hidden_dims, strides, filter_sizes, paddings,
            maxpool_filter_sizes, maxpool_strides, feature_dimension)
        self.online_projector = ProjectionNetwork(dimension, hidden_size, projection_size)
        self.online_predictor = PredictionNetwork(projection_size, hidden_size, projection_size)
        # setup target network
        self.target_encoder = None
        self.target_projector = None
        # loss function
        self.criterion = criterion.byol_criterion

    def get_target_ecnoder(self):
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, requires=False)

    def get_target_projector(self):
        self.target_projector = copy.deepcopy(self.online_projector)
        set_requires_grad(self.target_encoder, requires=False)

    def forward(self, x01, x02):
        if self.target_encoder is None or self.target_projector is None:
            self.get_target_ecnoder()
            self.get_target_projector()

        online_representation01 = self.online_encoder(x01)
        online_representation02 = self.online_encoder(x02)
        print(online_representation01.size())
        online_projection01 = self.online_projector(online_representation01)
        online_projection02 = self.online_projector(online_representation02)
        online_prediction01 = self.online_predictor(online_projection01)
        online_prediction02 = self.online_predictor(online_projection02)

        with torch.no_grad():
            target_representation01 = self.target_encoder(x01)
            target_representation02 = self.target_encoder(x02)
            target_projector01 = self.target_projector(target_representation01)
            target_projector02 = self.target_projector(target_representation02)

        loss01 = self.criterion(online_prediction01, target_projector02.detach())
        loss02 = self.criterion(online_prediction02, target_projector01.detach())
        loss = loss01 + loss02
        return online_representation01, loss.mean()



# dimensions of feature representation
class EncodingNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dims, strides, filter_sizes, paddings,
                 maxpool_filter_sizes, maxpool_strides, feature_dimension):
        super(EncodingNetwork, self).__init__()
        assert (
                len(strides) == len(filter_sizes) == len(paddings)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.linear_input_dimension = 64 * (64 // (2**3))
        self.feature_dimension = feature_dimension
        self.encoder = nn.Sequential()
        for index, (input_dim, hidden_dim, stride, filter_size, padding, maxpool_filter_size, maxpool_stride) in enumerate(
                zip(input_dims, hidden_dims, strides, filter_sizes, paddings, maxpool_filter_sizes, maxpool_strides)):
            self.encoder.add_module(
                "encoder_layer_{}".format(index),
                nn.Sequential(
                    nn.Conv2d(input_dim, hidden_dim, kernel_size=filter_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                    nn.MaxPool2d(maxpool_filter_size, stride=maxpool_stride),
                )
            )
        self.full_connected = nn.Sequential(
            nn.Linear(self.linear_input_dimension, feature_dimension),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(feature_dimension, feature_dimension),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.encoder(x) # (batch, ch, mel, time)
        out = out.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = out.shape
        out = out.reshape((B, T, C * D))
        out = self.full_connected(out)
        (out1, _) = torch.max(out, dim=1)
        out2 = torch.mean(out, dim=1)
        out = out1 + out2
        assert out.shape[1] == self.feature_dimension and out.ndim == 2
        return out


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


if __name__ == '__main__':
    model = BYOL(
        input_dims=[1, 64, 64],
        hidden_dims=[64, 64, 64],
        strides=[1, 1, 1],
        filter_sizes=[3, 3, 3],
        paddings=[1, 1, 1],
        maxpool_filter_sizes=[2, 2, 2],
        maxpool_strides=[2, 2, 2],
        feature_dimension=2048,
        hidden_size=256,
        projection_size=4096
    )

    input_data01 = torch.randn((2, 1, 64, 96))
    input_data02 = torch.randn((2, 1, 64, 96))
    output = model(input_data01, input_data02)
    print(output)



