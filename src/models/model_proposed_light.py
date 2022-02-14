import copy
import os
import collections
import torch
import torchvision
import torch.nn as nn
import src.losses.criterion as losses
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def set_requires_grad(model, requires):
    for parameter in model.parameters():
        parameter.requires_grad = requires


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, stride, filter_size, padding):
        super(Encoder, self).__init__()
        assert(
                len(stride) == len(filter_size) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.encoder = nn.Sequential()
        for index, (stride, filter_size, padding) in enumerate(zip(stride, filter_size, padding)):
            self.encoder.add_module(
                "encoder_layer_{}".format(index),
                nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                              kernel_size=filter_size, stride=stride, padding=padding),
                    nn.ReLU(),
                )
            )
            input_dim = hidden_dim

    def forward(self, x):
        return self.encoder(x)


class WaveBYOL(nn.Module):
    def __init__(self, config, encoder_input_dim, encoder_hidden_dim, encoder_filter_size, encoder_stride, encoder_padding,
                 mlp_input_dim, mlp_hidden_dim, mlp_output_dim):
        super(WaveBYOL, self).__init__()
        self.config = config
        self.online_encoder_network = Encoder(  # CPC encoder와 동일하게 매칭되는 부분
            input_dim=encoder_input_dim,
            hidden_dim=encoder_hidden_dim,
            filter_size=encoder_filter_size,
            stride=encoder_stride,
            padding=encoder_padding,
        )
        self.online_adavp_network = nn.AdaptiveAvgPool2d((512, 1))
        self.online_projector_network = MLPNetwork(mlp_input_dim, mlp_hidden_dim, mlp_output_dim)
        self.online_predictor_network = MLPNetwork(mlp_output_dim, mlp_hidden_dim, mlp_output_dim)

        # target network들은 나중에 online network꺼 그대로 가져다 쓸꺼니까 생성시점에서는 만들필요 없음
        self.target_encoder_network = None
        self.target_adavp_network = None
        self.target_projector_network = None

        # 아직도 이 loss에 대해서 좀 분분인데 일단은 그냥 쓰기로 햇음
        self.criterion = losses.byol_a_criterion

    def setup_target_network(self):
        self.get_encoder_network()
        self.get_adavp_network()
        self.get_target_projector()

    def get_encoder_network(self):
        self.target_encoder_network = copy.deepcopy(self.online_encoder_network)
        set_requires_grad(self.target_encoder_network, requires=False)

    def get_adavp_network(self):
        self.target_adavp_network = copy.deepcopy(self.online_adavp_network)
        set_requires_grad(self.target_adavp_network, requires=False)

    def get_target_projector(self):
        self.target_projector_network = copy.deepcopy(self.online_projector_network)
        set_requires_grad(self.target_projector_network, requires=False)

    def get_representation(self, x):
        pass

    def get_projection(self, x):
        pass

    def forward(self, x01, x02):
        if self.target_encoder_network is None or self.target_adavp_network is None \
                or self.target_projector_network is None:
            self.setup_target_network()

        online01 = self.online_encoder_network(x01)
        online01_adavp = self.online_adavp_network(online01)
        online01_adavp = online01_adavp.squeeze(2)
        online01_projection = self.online_projector_network(online01_adavp)
        online01_prediction = self.online_predictor_network(online01_projection)

        online02 = self.online_encoder_network(x02)
        online02_adavp = self.online_adavp_network(online02)
        online02_adavp = online02_adavp.squeeze(2)
        online02_projection = self.online_projector_network(online02_adavp)
        online02_prediction = self.online_predictor_network(online02_projection)

        with torch.no_grad():
            target01 = self.target_encoder_network(x01.detach())
            target01_adavp = self.target_adavp_network(target01)
            target01_adavp = target01_adavp.squeeze(2)
            target01_projection = self.target_projector_network(target01_adavp)

            target02 = self.target_encoder_network(x02.detach())
            target02_adavp = self.target_adavp_network(target02)
            target02_adavp = target02_adavp.squeeze(2)
            target02_projection = self.target_projector_network(target02_adavp)

        loss01 = self.criterion(online01_prediction, target02_projection.detach())
        loss02 = self.criterion(online02_prediction, target01_projection.detach())
        loss = loss01 + loss02
        loss = loss.mean()
        representations = [online01.detach(), online02.detach(), target01.detach(), target02.detach()]
        return loss, representations


if __name__ == '__main__':
    test_model = WaveBYOL(
        config=None,
        encoder_input_dim=1,
        encoder_hidden_dim=512,
        encoder_filter_sizes=[10, 8, 4, 4, 4],
        encoder_strides=[5, 4, 2, 2, 2],
        encoder_paddings=[2, 2, 2, 2, 1],
        mlp_input_dim=512,  # b4,15200 -> 86016
        mlp_hidden_dim=256,
        mlp_output_dim=4096,
    ).cuda()
    print(test_model)
    input_data01 = torch.rand(2, 1, 64000).cuda()
    input_data02 = torch.rand(2, 1, 64000).cuda()
    loss, outputs = test_model(input_data01, input_data02)
    print(loss)
    print(outputs[0].size())

