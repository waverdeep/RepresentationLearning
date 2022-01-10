import collections
import torch
import torch.nn as nn
import torchvision
import copy
import src.models.model_baseline as model_baseline
import src.losses.criterion as losses


# 모델 파라미터의 gradient 업데이트 여부를 결정
def set_requires_grad(model, requires):
    for parameter in model.parameters():
        parameter.requires_grad = requires


# 결과적으로 보면 pre-network와 encoder network 가 합쳐저야 하지 않나 싶음
class PreNetwork(model_baseline.Encoder):
    def __init__(self, input_dim, hidden_dim, strides, filter_sizes, paddings):
        super().__init__(input_dim, hidden_dim, strides, filter_sizes, paddings)


class EncoderNetwork(nn.Module):
    def __init__(self):
        super(EncoderNetwork, self).__init__()
        # 생각보다 vgg가 성능이 안좋지 않나 싶음
        # self.vgg16 = torchvision.models.vgg16_bn(pretrained=True, ).features
        self.network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer01", nn.Conv2d(1, 3, kernel_size=1, stride=1)), # it just 1*1 convolution
                    ("feature_extract_layer02", torchvision.models.resnet50(pretrained=False).conv1),
                    ("feature_extract_layer03", torchvision.models.resnet50(pretrained=False).bn1),
                    ("feature_extract_layer04", torchvision.models.resnet50(pretrained=False).relu),
                    ("feature_extract_layer05", torchvision.models.resnet50(pretrained=False).maxpool),
                    ("feature_extract_layer06", torchvision.models.resnet50(pretrained=False).layer1),
                    ("feature_extract_layer07", torchvision.models.resnet50(pretrained=False).layer2),
                    ("feature_extract_layer08", torchvision.models.resnet50(pretrained=False).layer3),
                    ("feature_extract_layer09", torchvision.models.resnet50(pretrained=False).layer4),

                ]
            )
        )

    def forward(self, x):
        out = self.network(x)
        return out


# projection network와 prediction network는 코드상으로 다른점이 하나도 없기 때문에 한번에 정의해도 됨
# 일부러 분리해서 작성했는데 구지 그럴 필요가 있었는지 이류를 찾는 중
class ProjectionNetwork(nn.Module):
    def __init__(self, dimension, hidden_size, projection_size):
        super(ProjectionNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.network(x)


class PredictionNetwork(nn.Module):
    def __init__(self, dimension, hidden_size, prediction_size):
        super(PredictionNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, prediction_size),
        )

    def forward(self, x):
        return self.network(x)


class WaveBYOL(nn.Module):
    def __init__(self, config, pre_input_dims, pre_hidden_dims, pre_filter_sizes, pre_strides, pre_paddings,
                 dimension, hidden_size, projection_size):
        super(WaveBYOL, self).__init__()
        self.config = config
        self.online_pre_network = PreNetwork( # CPC encoder와 동일하게 매칭되는 부분
            input_dim=pre_input_dims,
            hidden_dim=pre_hidden_dims,
            filter_sizes=pre_filter_sizes,
            strides=pre_strides,
            paddings=pre_paddings,
        )
        # audio_window가 달라진다면 바뀌어야할 파라미터들이 있어서, 나중에는 파라미터로 직접 받을 수 있게 변경하는 것이 좋을 듯
        self.online_encoder_network = EncoderNetwork()
        self.online_projector_network = ProjectionNetwork(dimension, hidden_size, projection_size)
        self.online_predictor_network = PredictionNetwork(projection_size, hidden_size, projection_size)

        # target network들은 나중에 online network꺼 그대로 가져다 쓸꺼니까 생성시점에서는 만들필요 없음
        self.target_pre_network = None
        self.target_encoder_network = None
        self.target_projector_network = None

        # 아직도 이 loss에 대해서 좀 분분인데 일단은 그냥 쓰기로 햇음
        self.criterion = losses.byol_a_criterion

    def get_pre_network(self):
        self.target_pre_network = copy.deepcopy(self.online_pre_network)
        set_requires_grad(self.target_pre_network, requires=False)

    def get_target_encoder(self):
        self.target_encoder_network = copy.deepcopy(self.online_encoder_network)
        set_requires_grad(self.target_encoder_network, requires=False)

    def get_target_projector(self):
        self.target_projector_network = copy.deepcopy(self.online_projector_network)
        set_requires_grad(self.target_projector_network, requires=False)

    def get_representation(self, x):
        online_x = self.online_pre_network(x)
        online_x = online_x.unsqueeze(1)
        online_representation = self.online_encoder_network(online_x)
        return online_representation

    def forward(self, x01, x02):
        # 먼저 target network 파라미터부터 따와서 생성
        if self.target_pre_network is None \
                or self.target_encoder_network is None or self.target_projector_network is None:
            self.get_pre_network()
            self.get_target_encoder()
            self.get_target_projector()

        # online network 관련 코드부터 실행 (x01과 x02 모두)
        # input: (batch, frequency, timestep)
        # output: (batch, frequency, timestep)
        online_x01 = self.online_pre_network(x01)
        online_x02 = self.online_pre_network(x02)
        # shape: (batch, channel, frequency, timestep)
        online_x01 = online_x01.unsqueeze(1)
        online_x02 = online_x02.unsqueeze(1)
        # input: (batch, channel, frequency, timestep)
        # output: (batch, channel, frequency, timestep) -> 여기서 channel 이 빵빵해지고 나머지가 줄어들 것
        online_representation01 = self.online_encoder_network(online_x01)
        online_representation02 = self.online_encoder_network(online_x02)

        # shape 변경: (batch, time, frequency (mel), channel)
        online_representation01_reshape = online_representation01.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        online_representation02_reshape = online_representation02.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        B1, T1, D1, C1 = online_representation01_reshape.shape
        B2, T2, D2, C2 = online_representation02_reshape.shape
        # shape 변경 (batch, time, frequency * channel)
        online_representation01_reshape = online_representation01_reshape.reshape((B1, T1 * C1 * D1))
        online_representation02_reshape = online_representation02_reshape.reshape((B2, T2 * C2 * D2))
        # ** projection과 prediction들어가기 전에 한번더 변환해주어야 함 (아니면 투딤으로 그냥 가버려?)
        print(online_representation01_reshape.size())
        online_projection01 = self.online_projector_network(online_representation01_reshape)
        online_projection02 = self.online_projector_network(online_representation02_reshape)
        online_prediction01 = self.online_predictor_network(online_projection01)
        online_prediction02 = self.online_predictor_network(online_projection02)

        with torch.no_grad():
            # input: (batch, frequency, timestep)
            # output: (batch, frequency, timestep)
            target_x01 = self.target_pre_network(x01)
            target_x02 = self.target_pre_network(x02)
            # shape: (batch, channel, frequency, timestep)
            target_x01 = target_x01.unsqueeze(1)
            target_x02 = target_x02.unsqueeze(1)
            # input: (batch, channel, frequency, timestep)
            # output: (batch, channel, frequency, timestep) -> 여기서 channel 이 빵빵해지고 나머지가 줄어들 것
            target_representation01 = self.target_encoder_network(target_x01)
            target_representation02 = self.target_encoder_network(target_x02)
            # shape 변경: (batch, time, frequency (mel), channel)
            target_representation01_reshape = target_representation01.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
            target_representation02_reshape = target_representation02.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
            B1, T1, D1, C1 = target_representation01_reshape.shape
            B2, T2, D2, C2 = target_representation02_reshape.shape
            # shape 변경 (batch, time, frequency * channel)
            target_representation01_reshape = target_representation01_reshape.reshape((B1, T1 * C1 * D1))
            target_representation02_reshape = target_representation02_reshape.reshape((B2, T2 * C2 * D2))
            print(target_representation02_reshape.size())
            # target line은  projection만 시킨다~
            target_projection01 = self.target_projector_network(target_representation01_reshape)
            target_projection02 = self.target_projector_network(target_representation02_reshape)

        # 정말 loss 구하는 공식이 맞는지 잘 모르겠지만 일단 해본다
        # detach는 gradient 안딸려오게 복사하는 것~~
        loss01 = self.criterion(online_prediction01, target_projection02.detach())
        loss02 = self.criterion(online_prediction02, target_projection01.detach())
        loss = loss01 + loss02
        return online_representation01, loss.mean()


if __name__ == '__main__':
    data = torch.rand(8, 1, 20480)
    # pre_network = PreNetwork(
    #     input_dim=1,
    #     hidden_dim=512,
    #     filter_sizes=[10, 8, 4, 4, 4],
    #     strides=[5, 4, 2, 2, 2],
    #     paddings=[2, 2, 2, 2, 1],
    # )
    # out01 = pre_network(data)
    # print(out01.size())
    #
    # out02 = out01.unsqueeze(1)
    # print(out02.size())
    #
    # encoder_network = EncoderNetwork()
    # out03 = encoder_network(out02)
    # print(out03.size())
    # #
    # projector = ProjectionNetwork(131072, 512, 2048)
    # predictor = PredictionNetwork(2048, 512, 2048)
    #
    # toy_data = torch.rand(8, 1, 20480)
    # out = pre_network(toy_data)
    # print(out.size()) # torch.Size([8, 512, 128])
    # #
    # out = out.unsqueeze(1)
    # print(out.size())
    # #
    # out = encoder_network(out)
    # print(out.size())
    # #
    # out = out.permute(0, 3, 2, 1)
    # print(out.size())
    # #
    # B, T, D, C = out.shape
    # out = out.reshape((B, T * C * D))
    # print(out.size()) # torch.Size([8, 4, 32768]) for resnet50
    #
    # out = projector(out)
    # print(out.size())
    #
    # out = predictor(out)
    # print(out.size())

    test_model = WaveBYOL(
        config=None,
        pre_input_dims=1,
        pre_hidden_dims=512,
        pre_filter_sizes=[10, 8, 4, 4, 4],
        pre_strides=[5, 4, 2, 2, 2],
        pre_paddings=[2, 2, 2, 2, 1],
        dimension=131072,
        hidden_size=256,
        projection_size=4096
    ).cuda()
    print(test_model)

    input_data01 = torch.rand(8, 1, 20480).cuda()
    input_data02 = torch.rand(8, 1, 20480).cuda()
    output, _ = test_model(input_data01, input_data02)
    print(output.size())





