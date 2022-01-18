import torch
import copy
import os
import torch.nn as nn
import collections
import torchvision
from efficientnet_pytorch import EfficientNet
import src.models.model_proposed02 as model_proposed02
import src.models.model_proposed05 as model_proposed05
import src.losses.criterion as losses
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class EfficientBYOL(nn.Module):
    def __init__(self, config, pre_input_dims, pre_hidden_dims, pre_filter_sizes, pre_strides, pre_paddings,
                 dimension, hidden_size, projection_size, research=True):
        super(EfficientBYOL, self).__init__()
        self.config = config
        self.research = research
        self.online_pre_network = model_proposed02.PreNetwork(  # CPC encoder와 동일하게 매칭되는 부분
            input_dim=pre_input_dims,
            hidden_dim=pre_hidden_dims,
            filter_sizes=pre_filter_sizes,
            strides=pre_strides,
            paddings=pre_paddings,
        )
        self.online_encoder_network = model_proposed05.EncoderNetwork()
        self.online_projector_network = model_proposed02.ProjectionNetwork(dimension, hidden_size, projection_size)
        self.online_predictor_network = model_proposed02.PredictionNetwork(projection_size, hidden_size, projection_size)

        # target network들은 나중에 online network꺼 그대로 가져다 쓸꺼니까 생성시점에서는 만들필요 없음
        self.target_pre_network = None
        self.encoder_network = None
        self.target_projector_network = None

        # modest network
        self.modest_pre_network = None
        self.modest_encoder_network = None
        self.modest_projector_network = None

        # 아직도 이 loss에 대해서 좀 분분인데 일단은 그냥 쓰기로 햇음
        self.criterion = losses.set_criterion("MSELoss")
        self.output_representation = nn.AdaptiveAvgPool3d((1, 16, 4))

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

    def setup_modest_network(self):
        self.get_pre_network()
        self.get_modest_encoder()
        self.get_modest_projector()

    def get_modest_pre_network(self):
        self.modest_pre_network = copy.deepcopy(self.online_pre_network)
        model_proposed02.set_requires_grad(self.modest_pre_network, requires=False)

    def get_modest_encoder(self):
        self.modest_encoder_network = copy.deepcopy(self.online_encoder_network)
        model_proposed02.set_requires_grad(self.modest_encoder_network, requires=False)

    def get_modest_projector(self):
        self.modest_projector_network = copy.deepcopy(self.online_projector_network)
        model_proposed02.set_requires_grad(self.modest_projector_network, requires=False)

    def get_representation(self, x):
        output = self.online_pre_network(x)
        output = output.unsqueeze(1)
        online_representation = self.online_encoder_network(output)
        return online_representation

    def forward(self, x01, x02, x03):
        # 먼저 target network 파라미터부터 따와서 생성
        if self.target_pre_network is None \
                or self.target_encoder_network is None or self.target_projector_network is None:
            self.get_pre_network()
            self.get_target_encoder()
            self.get_target_projector()

        if self.modest_pre_network is None \
                or self.modest_encoder_network is None or self.modest_projector_network is None:
            self.get_modest_pre_network()
            self.get_modest_encoder()
            self.get_modest_projector()

        online_x01_pre = self.online_pre_network(x01)
        online_x02_pre = self.online_pre_network(x02)
        online_x03_pre = self.online_pre_network(x03)

        online_x01 = online_x01_pre.unsqueeze(1)
        online_x02 = online_x02_pre.unsqueeze(1)
        online_x03 = online_x03_pre.unsqueeze(1)

        online_representation01 = self.online_encoder_network(online_x01)
        online_representation02 = self.online_encoder_network(online_x02)
        online_representation03 = self.online_encoder_network(online_x03)

        online_representation01_output = self.output_representation(online_representation01)
        online_representation02_output = self.output_representation(online_representation02)
        online_representation03_output = self.output_representation(online_representation03)

        online_representation01_reshape = online_representation01.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        online_representation02_reshape = online_representation02.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        online_representation03_reshape = online_representation03.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        B1, T1, D1, C1 = online_representation01_reshape.shape
        B2, T2, D2, C2 = online_representation02_reshape.shape
        B3, T3, D3, C3 = online_representation03_reshape.shape

        online_representation01_reshape = online_representation01_reshape.reshape((B1, T1 * C1 * D1))
        online_representation02_reshape = online_representation02_reshape.reshape((B2, T2 * C2 * D2))
        online_representation03_reshape = online_representation02_reshape.reshape((B3, T3 * C3 * D3))

        online_projection01 = self.online_projector_network(online_representation01_reshape)
        online_projection02 = self.online_projector_network(online_representation02_reshape)
        online_projection03 = self.online_projector_network(online_representation03_reshape)

        online_prediction01 = self.online_predictor_network(online_projection01)
        online_prediction02 = self.online_predictor_network(online_projection02)
        online_prediction03 = self.online_predictor_network(online_projection03)

        with torch.no_grad():
            target_x01_pre = self.target_pre_network(x01)
            target_x02_pre = self.target_pre_network(x02)

            target_x01 = target_x01_pre.unsqueeze(1)
            target_x02 = target_x02_pre.unsqueeze(1)

            target_representation01 = self.target_encoder_network(target_x01)
            target_representation02 = self.target_encoder_network(target_x02)

            target_representation01_output = self.output_representation(target_representation01)
            target_representation02_output = self.output_representation(target_representation02)

            target_representation01_reshape = target_representation01.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
            target_representation02_reshape = target_representation02.permute(0, 3, 2, 1)  # (batch, time, mel, ch)

            B1, T1, D1, C1 = target_representation01_reshape.shape
            B2, T2, D2, C2 = target_representation02_reshape.shape

            target_representation01_reshape = target_representation01_reshape.reshape((B1, T1 * C1 * D1))
            target_representation02_reshape = target_representation02_reshape.reshape((B2, T2 * C2 * D2))

            target_projection01 = self.target_projector_network(target_representation01_reshape)
            target_projection02 = self.target_projector_network(target_representation02_reshape)

        with torch.no_grad():
            modest_x01_pre = self.modest_pre_network(x01)
            modest_x03_pre = self.modest_pre_network(x03)

            modest_x01 = modest_x01_pre.unsqueeze(1)
            modest_x03 = modest_x03_pre.unsqueeze(1)

            modest_representation01 = self.modest_encoder_network(modest_x01)
            modest_representation03 = self.modest_encoder_network(modest_x03)

            modest_representation01_output = self.output_representation(modest_representation01)
            modest_representation03_output = self.output_representation(modest_representation03)

            modest_representation01_reshape = modest_representation01.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
            modest_representation03_reshape = modest_representation03.permute(0, 3, 2, 1)  # (batch, time, mel, ch)

            B1, T1, D1, C1 = modest_representation01_reshape.shape
            B2, T2, D2, C2 = modest_representation03_reshape.shape

            modest_representation01_reshape = modest_representation01_reshape.reshape((B1, T1 * C1 * D1))
            modest_representation03_reshape = modest_representation03_reshape.reshape((B3, T3 * C3 * D3))

            modest_projection01 = self.target_projector_network(modest_representation01_reshape)
            modest_projection03 = self.target_projector_network(modest_representation03_reshape)

        target_loss01 = self.criterion(online_prediction01, target_projection02.detach())
        target_loss02 = self.criterion(online_prediction02, target_projection01.detach())
        modest_loss01 = self.criterion(online_prediction01, modest_projection03.detach())
        modest_loss02 = self.criterion(online_prediction03, modest_projection01.detach())

        loss = target_loss01 + target_loss02 + modest_loss01 + modest_loss02
        return online_x01_pre, online_x02_pre, online_representation01_output, online_representation02_output, \
               target_x01_pre, target_x02_pre, target_representation01_output, target_representation02_output, \
               loss.mean()


if __name__ == '__main__':
    test_model = EfficientBYOL(
        config=None,
        pre_input_dims=1,
        pre_hidden_dims=512,
        pre_filter_sizes=[10, 8, 4, 4, 4],
        pre_strides=[5, 4, 2, 2, 2],
        pre_paddings=[2, 2, 2, 2, 1],
        dimension=163840,
        hidden_size=256,
        projection_size=4096
    ).cuda()
    print(test_model)

    input_data01 = torch.rand(8, 1, 20480).cuda()
    input_data02 = torch.rand(8, 1, 20480).cuda()
    input_data03 = torch.rand(8, 1, 20480).cuda()
    online_x01_pre, online_x02_pre, online_representation01_output, online_representation02_output, \
    target_x01_pre, target_x02_pre, target_representation01_output, target_representation02_output, \
    loss = test_model(input_data01, input_data02, input_data03)
    print(online_x01_pre.size())
    print(online_representation01_output.size())