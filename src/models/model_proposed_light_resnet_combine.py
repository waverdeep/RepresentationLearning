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


class PostNetwork(nn.Module):
    def __init__(self):
        super(PostNetwork, self).__init__()
        self.pooling_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_layer = nn.Flatten()

    def forward(self, x):
        out = self.pooling_layer(x)
        out = self.flatten_layer(out)
        return out


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, stride, filter_size, padding, resnet_version):
        super(Encoder, self).__init__()
        assert(
                len(stride) == len(filter_size) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.encoder01 = nn.Sequential()
        for index, (stride, filter_size, padding) in enumerate(zip(stride, filter_size, padding)):
            self.encoder01.add_module(
                "cpc_encoder_layer_{}".format(index),
                nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                              kernel_size=filter_size, stride=stride, padding=padding),
                    nn.ReLU(),
                )
            )
            input_dim = hidden_dim
        self.encoder02 = nn.Sequential()
        if resnet_version == "18":
            self.encoder02.add_module(
                "resnet_encoder_layer",
                nn.Sequential(
                    # 원래 resnet은 3 to 64 인데, 여기서는 1채널부터 시작하기 때문에 1 to 64로
                    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    torchvision.models.resnet18(pretrained=True).bn1,
                    torchvision.models.resnet18(pretrained=True).relu,
                    # maxpool layer는 사용하지 않고 작업을 해보려고 함.
                    torchvision.models.resnet18(pretrained=True).layer1,
                    torchvision.models.resnet18(pretrained=True).layer2,
                    torchvision.models.resnet18(pretrained=True).layer3,
                    torchvision.models.resnet18(pretrained=True).layer4
                )
            )
        elif resnet_version == "50":
            self.encoder02.add_module(
                "resnet_encoder_layer",
                nn.Sequential(
                    # 원래 resnet은 3 to 64 인데, 여기서는 1채널부터 시작하기 때문에 1 to 64로
                    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    torchvision.models.resnet50(pretrained=True).bn1,
                    torchvision.models.resnet50(pretrained=True).relu,
                    # maxpool layer는 사용하지 않고 작업을 해보려고 함.
                    torchvision.models.resnet50(pretrained=True).layer1,
                    torchvision.models.resnet50(pretrained=True).layer2,
                    torchvision.models.resnet50(pretrained=True).layer3,
                    torchvision.models.resnet50(pretrained=True).layer4
                )
            )
        elif resnet_version == "50_2":
            self.encoder02.add_module(
                "resnet_encoder_layer",
                nn.Sequential(
                    # 원래 resnet은 3 to 64 인데, 여기서는 1채널부터 시작하기 때문에 1 to 64로
                    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    torchvision.models.wide_resnet50_2(pretrained=True).bn1,
                    torchvision.models.wide_resnet50_2(pretrained=True).relu,
                    # maxpool layer는 사용하지 않고 작업을 해보려고 함.
                    torchvision.models.wide_resnet50_2(pretrained=True).layer1,
                    torchvision.models.wide_resnet50_2(pretrained=True).layer2,
                    torchvision.models.wide_resnet50_2(pretrained=True).layer3,
                    torchvision.models.wide_resnet50_2(pretrained=True).layer4
                )
            )
        elif resnet_version == "101_2":
            self.encoder02.add_module(
                "resnet_encoder_layer",
                nn.Sequential(
                    # 원래 resnet은 3 to 64 인데, 여기서는 1채널부터 시작하기 때문에 1 to 64로
                    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    torchvision.models.wide_resnet101_2(pretrained=True).bn1,
                    torchvision.models.wide_resnet101_2(pretrained=True).relu,
                    # maxpool layer는 사용하지 않고 작업을 해보려고 함.
                    torchvision.models.wide_resnet101_2(pretrained=True).layer1,
                    torchvision.models.wide_resnet101_2(pretrained=True).layer2,
                    torchvision.models.wide_resnet101_2(pretrained=True).layer3,
                    torchvision.models.wide_resnet101_2(pretrained=True).layer4
                )
            )
        elif resnet_version == "152":
            self.encoder02.add_module(
                "resnet_encoder_layer",
                nn.Sequential(
                    # 원래 resnet은 3 to 64 인데, 여기서는 1채널부터 시작하기 때문에 1 to 64로
                    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    torchvision.models.resnet152(pretrained=False).bn1,
                    torchvision.models.resnet152(pretrained=False).relu,
                    # maxpool layer는 사용하지 않고 작업을 해보려고 함.
                    torchvision.models.resnet152(pretrained=False).layer1,
                    torchvision.models.resnet152(pretrained=False).layer2,
                    torchvision.models.resnet152(pretrained=False).layer3,
                    torchvision.models.resnet152(pretrained=False).layer4
                )
            )

    def forward(self, x):
        out = self.encoder01(x)
        out = torch.transpose(out, 1, 2)
        out = out.unsqueeze(1)
        out = self.encoder02(out)
        return out


class WaveBYOL(nn.Module):
    def __init__(self, config, encoder_input_dim, encoder_hidden_dim, encoder_filter_size, encoder_stride, encoder_padding,
                 mlp_input_dim, mlp_hidden_dim, mlp_output_dim, resnet_version="50"):
        super(WaveBYOL, self).__init__()
        self.config = config
        self.online_encoder_network = Encoder(  # CPC encoder와 동일하게 매칭되는 부분
            input_dim=encoder_input_dim,
            hidden_dim=encoder_hidden_dim,
            filter_size=encoder_filter_size,
            stride=encoder_stride,
            padding=encoder_padding,
            resnet_version=resnet_version
        )
        self.online_post_network = PostNetwork()
        self.online_projector_network = MLPNetwork(mlp_input_dim, mlp_hidden_dim, mlp_output_dim)
        self.online_predictor_network = MLPNetwork(mlp_output_dim, mlp_hidden_dim, mlp_output_dim)


        # target network들은 나중에 online network꺼 그대로 가져다 쓸꺼니까 생성시점에서는 만들필요 없음
        self.target_encoder_network = None
        self.target_post_network = None
        self.target_projector_network = None

        # 아직도 이 loss에 대해서 좀 분분인데 일단은 그냥 쓰기로 햇음
        self.criterion = losses.byol_a_criterion

    def setup_target_network(self):
        self.get_encoder_network()
        self.get_post_network()
        self.get_target_projector()

    def get_encoder_network(self):
        self.target_encoder_network = None
        self.target_encoder_network = copy.deepcopy(self.online_encoder_network)
        set_requires_grad(self.target_encoder_network, requires=False)

    def get_post_network(self):
        self.target_post_network = None
        self.target_post_network = copy.deepcopy(self.online_post_network)
        set_requires_grad(self.target_post_network, requires=False)

    def get_target_projector(self):
        self.target_projector_network = None
        self.target_projector_network = copy.deepcopy(self.online_projector_network)
        set_requires_grad(self.target_projector_network, requires=False)

    def get_representation(self, x):
        pass

    def get_projection(self, x):
        pass

    def forward(self, x01, x02):
        if self.target_encoder_network is None or self.target_post_network is None \
                or self.target_projector_network is None:
            self.setup_target_network()

        online01 = self.online_encoder_network(x01)
        online01_post = self.online_post_network(online01)
        online01_projection = self.online_projector_network(online01_post)
        online01_prediction = self.online_predictor_network(online01_projection)

        online02 = self.online_encoder_network(x02)
        online02_post = self.online_post_network(online02)
        online02_projection = self.online_projector_network(online02_post)
        online02_prediction = self.online_predictor_network(online02_projection)

        with torch.no_grad():
            target01 = self.target_encoder_network(x01.detach())
            target01_post = self.target_post_network(target01)
            target01_projection = self.target_projector_network(target01_post)

            target02 = self.target_encoder_network(x02.detach())
            target02_post = self.target_post_network(target02)
            target02_projection = self.target_projector_network(target02_post)

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
        encoder_filter_size=[10, 8, 4, 4],
        encoder_stride=[5, 4, 2, 2],
        encoder_padding=[2, 2, 2, 1],
        mlp_input_dim=2048,
        mlp_hidden_dim=256,
        mlp_output_dim=4096,
        resnet_version="50_2"
    ).cuda()
    print(test_model)
    input_data01 = torch.rand(2, 1, 20480).cuda()
    input_data02 = torch.rand(2, 1, 20480).cuda()
    loss, outputs = test_model(input_data01, input_data02)
    print(loss)
    print(outputs[0].size())

