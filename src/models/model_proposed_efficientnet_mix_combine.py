import copy
import os
import collections
import torch
import torchvision
import torch.nn as nn
import src.losses.criterion as losses
import src.models.model_proposed02 as model_proposed02
from efficientnet_pytorch import EfficientNet
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
                              kernel_size=filter_size, stride=stride, padding=padding),
                    nn.LeakyReLU(),
                )
            )
            input_dim = hidden_dim

    def forward(self, x):
        return self.encoder(x)


class EncoderNetwork(nn.Module):
    def __init__(self, efficientnet_model_name='efficientnet-b4'):
        super(EncoderNetwork, self).__init__()
        self.network = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature_extract_layer01", nn.Conv2d(1, 3, kernel_size=1, stride=1)),  # it just 1*1 convolution
                ]
            )
        )
        self.efficient_network = EfficientNet.from_pretrained(efficientnet_model_name)

    def forward(self, x):
        out = self.network(x)
        out = self.efficient_network.extract_features(out)
        return out


class WaveBYOLEfficient(nn.Module):
    def __init__(self, config, pre_input_dims, pre_hidden_dims, pre_filter_sizes, pre_strides, pre_paddings,
                 dimension, hidden_size, projection_size, efficientnet_model_name):
        super(WaveBYOLEfficient, self).__init__()
        self.config = config
        self.online_pre_network = PreNetwork(  # CPC encoder와 동일하게 매칭되는 부분 -> leakyrelu로 변경
            input_dim=pre_input_dims,
            hidden_dim=pre_hidden_dims,
            filter_sizes=pre_filter_sizes,
            strides=pre_strides,
            paddings=pre_paddings,
        )
        self.online_encoder_network = EncoderNetwork(efficientnet_model_name=efficientnet_model_name)
        self.online_projector_network = model_proposed02.ProjectionNetwork(dimension, hidden_size, projection_size)
        self.online_predictor_network = model_proposed02.PredictionNetwork(projection_size, hidden_size, projection_size)

        # target network들은 나중에 online network꺼 그대로 가져다 쓸꺼니까 생성시점에서는 만들필요 없음
        self.target_pre_network = None
        self.encoder_network = None
        self.target_projector_network = None

        # 아직도 이 loss에 대해서 좀 분분인데 일단은 그냥 쓰기로 햇음
        self.criterion = losses.byol_a_criterion
        self.output_representation = nn.AdaptiveAvgPool3d((1, 16, 1024))

    def setup_target_network(self):
        self.get_pre_network()
        self.get_target_encoder()
        self.get_target_projector()

    def get_pre_network(self):
        self.target_pre_network = copy.deepcopy(self.online_pre_network)
        model_proposed02.set_requires_grad(self.target_pre_network, requires=False)

    def get_target_encoder(self):
        self.target_encoder_network = copy.deepcopy(self.online_encoder_network)
        model_proposed02.set_requires_grad(self.target_encoder_network, requires=False)

    def get_target_projector(self):
        self.target_projector_network = copy.deepcopy(self.online_projector_network)
        model_proposed02.set_requires_grad(self.target_projector_network, requires=False)

    def get_representation(self, x):
        output = self.online_pre_network(x)
        output = output.unsqueeze(1)
        online_representation = self.online_encoder_network(output)
        online_representation_reshape = online_representation.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        online_representation_output = self.output_representation(online_representation_reshape)
        return online_representation_output

    def get_projection(self, x):
        online_representation_output = self.get_representation(x)
        B1, T1, D1, C1 = online_representation_output.shape
        online_representation_reshape = online_representation_output.reshape((B1, T1 * C1 * D1))
        return online_representation_reshape, 0


    def forward(self, x01, x02):
        print(x01.size())
        # 먼저 target network 파라미터부터 따와서 생성
        if self.target_pre_network is None \
                or self.target_encoder_network is None or self.target_projector_network is None:
            self.get_pre_network()
            self.get_target_encoder()
            self.get_target_projector()

        # online network 관련 코드부터 실행 (x01과 x02 모두)
        # input: (batch, frequency, timestep)
        # output: (batch, frequency, timestep)
        online_x01_pre = self.online_pre_network(x01)
        print(online_x01_pre.size())
        online_x02_pre = self.online_pre_network(x02)
        # shape: (batch, channel, frequency, timestep)
        online_x01 = online_x01_pre.unsqueeze(1)
        print(online_x01.size())
        online_x02 = online_x02_pre.unsqueeze(1)
        # input: (batch, channel, frequency, timestep)
        # output: (batch, channel, frequency, timestep) -> 여기서 channel 이 빵빵해지고 나머지가 줄어들 것
        online_representation01 = self.online_encoder_network(online_x01)
        print(online_representation01.size())
        online_representation02 = self.online_encoder_network(online_x02)

        online_representation01_reshape = online_representation01.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        print(online_representation01_reshape.size())
        online_representation02_reshape = online_representation02.permute(0, 3, 2, 1)  # (batch, time, mel, ch)

        online_representation01_output = self.output_representation(online_representation01_reshape)
        print(online_representation01_output.size())
        online_representation02_output = self.output_representation(online_representation02_reshape)

        B1, T1, D1, C1 = online_representation01_output.shape
        B2, T2, D2, C2 = online_representation02_output.shape
        # shape 변경 (batch, time, frequency * channel)
        online_representation01_reshape = online_representation01_output.reshape((B1, T1 * C1 * D1))
        print(online_representation01_reshape.size())
        online_representation02_reshape = online_representation02_output.reshape((B2, T2 * C2 * D2))

        # ** projection과 prediction들어가기 전에 한번더 변환해주어야 함 (아니면 투딤으로 그냥 가버려?)
        # print(online_representation02_reshape.size())
        online_projection01 = self.online_projector_network(online_representation01_reshape)
        online_projection02 = self.online_projector_network(online_representation02_reshape)
        online_prediction01 = self.online_predictor_network(online_projection01)
        online_prediction02 = self.online_predictor_network(online_projection02)
        # print("projection output : ", online_projection01.size())

        with torch.no_grad():
            # input: (batch, frequency, timestep)
            # output: (batch, frequency, timestep)
            target_x01_pre = self.target_pre_network(x01)
            target_x02_pre = self.target_pre_network(x02)
            # shape: (batch, channel, frequency, timestep)
            target_x01 = target_x01_pre.unsqueeze(1)
            target_x02 = target_x02_pre.unsqueeze(1)
            # input: (batch, channel, frequency, timestep)
            # output: (batch, channel, frequency, timestep) -> 여기서 channel 이 빵빵해지고 나머지가 줄어들 것
            target_representation01 = self.target_encoder_network(target_x01)
            target_representation02 = self.target_encoder_network(target_x02)
            target_representation01_output = self.output_representation(target_representation01)
            target_representation02_output = self.output_representation(target_representation02)
            # shape 변경: (batch, time, frequency (mel), channel)
            target_representation01_reshape = target_representation01_output.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
            target_representation02_reshape = target_representation02_output.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
            B1, T1, D1, C1 = target_representation01_reshape.shape
            B2, T2, D2, C2 = target_representation02_reshape.shape
            # shape 변경 (batch, time, frequency * channel)
            target_representation01_reshape = target_representation01_reshape.reshape((B1, T1 * C1 * D1))
            target_representation02_reshape = target_representation02_reshape.reshape((B2, T2 * C2 * D2))
            # target line은  projection만 시킨다~
            target_projection01 = self.target_projector_network(target_representation01_reshape)
            target_projection02 = self.target_projector_network(target_representation02_reshape)

        # 정말 loss 구하는 공식이 맞는지 잘 모르겠지만 일단 해본다
        # detach는 gradient 안딸려오게 복사하는 것~~
        loss01 = self.criterion(online_prediction01, target_projection02.detach())
        loss02 = self.criterion(online_prediction02, target_projection01.detach())
        loss = loss01 + loss02
        online_representation = [(online_x01_pre, online_x02_pre,), (online_representation01_output, online_representation02_output,)]
        target_representation = [(target_x01_pre, target_x02_pre,), (target_representation01_output, target_representation02_output,)]
        return online_representation, target_representation, loss.mean()


if __name__ == '__main__':
    model_type = 'b4'
    if model_type == 'b7':
        test_model = WaveBYOLEfficient(
            config=None,
            pre_input_dims=1,
            pre_hidden_dims=512,
            pre_filter_sizes=[10, 8, 4, 4, 4],
            pre_strides=[5, 4, 2, 2, 2],
            pre_paddings=[2, 2, 2, 2, 1],
            dimension=64,  # b4,15200 -> 86016
            hidden_size=256,
            projection_size=4096,
            efficientnet_model_name='efficientnet-b7'
        ).cuda()
        print(test_model)
        input_data01 = torch.rand(2, 1, 20480).cuda()
        input_data02 = torch.rand(2, 1, 20480).cuda()
        online_representation_output, target_representation_output, loss = test_model(input_data01, input_data02)
        print(online_representation_output[0][0].size())
        print(online_representation_output[1][0].size())
    elif model_type == 'b0':
        test_model = WaveBYOLEfficient(
            config=None,
            pre_input_dims=1,
            pre_hidden_dims=512,
            pre_filter_sizes=[10, 8, 4, 4, 4],
            pre_strides=[5, 4, 2, 2, 2],
            pre_paddings=[2, 2, 2, 2, 1],
            dimension=16384,  # b4,15200 -> 86016
            hidden_size=2048,
            projection_size=4096,
            efficientnet_model_name='efficientnet-b0'
        ).cuda()
        print(test_model)
        input_data01 = torch.rand(2, 1, 20480).cuda()
        input_data02 = torch.rand(2, 1, 20480).cuda()
        online_representation_output, target_representation_output, loss = test_model(input_data01, input_data02)
        print(online_representation_output[0][0].size())
        print(online_representation_output[1][0].size())
    elif model_type == 'b4':
        test_model = WaveBYOLEfficient(
            config=None,
            pre_input_dims=1,
            pre_hidden_dims=512,
            pre_filter_sizes=[10, 8, 4, 4, 4],
            pre_strides=[5, 4, 2, 2, 2],
            pre_paddings=[2, 2, 2, 2, 1],
            dimension=16384,  # b4,15200 -> 86016
            hidden_size=2048,
            projection_size=4096,
            efficientnet_model_name='efficientnet-b4'
        ).cuda()
        # print(test_model)
        input_data01 = torch.rand(2, 1, 64000).cuda()
        input_data02 = torch.rand(2, 1, 15200).cuda()
        # online_representation_output, target_representation_output, loss = test_model(input_data01, input_data02)
        # print(online_representation_output[0][0].size())
        # print(online_representation_output[1][0].size())

        # rep, vec = test_model.get_projection(input_data01)
        # print(rep.size())
        # print(vec[0].size())
        # print(vec[1].size())


        rep, _ = test_model.get_projection(input_data01)
        print(rep.size())