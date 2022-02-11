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


class WaveBYOL(nn.Module):
    def __init__(self, config, pre_input_dims, pre_hidden_dims, pre_filter_sizes, pre_strides, pre_paddings,
                 dimension, hidden_size, projection_size):
        super(WaveBYOL, self).__init__()
        self.config = config
        self.online_pre_network = model_proposed02.PreNetwork(  # CPC encoder와 동일하게 매칭되는 부분
            input_dim=pre_input_dims,
            hidden_dim=pre_hidden_dims,
            filter_sizes=pre_filter_sizes,
            strides=pre_strides,
            paddings=pre_paddings,
        )
        self.online_projector_network = model_proposed02.ProjectionNetwork(dimension, hidden_size, projection_size)
        self.online_predictor_network = model_proposed02.PredictionNetwork(projection_size, hidden_size, projection_size)

        # target network들은 나중에 online network꺼 그대로 가져다 쓸꺼니까 생성시점에서는 만들필요 없음
        self.target_pre_network = None
        self.target_projector_network = None

        # 아직도 이 loss에 대해서 좀 분분인데 일단은 그냥 쓰기로 햇음
        self.criterion = losses.byol_a_criterion

    def setup_target_network(self):
        self.get_pre_network()
        self.get_target_projector()

    def get_pre_network(self):
        self.target_pre_network = copy.deepcopy(self.online_pre_network)
        model_proposed02.set_requires_grad(self.target_pre_network, requires=False)

    def get_target_projector(self):
        self.target_projector_network = copy.deepcopy(self.online_projector_network)
        model_proposed02.set_requires_grad(self.target_projector_network, requires=False)

    def get_representation(self, x):
        output = self.online_pre_network(x)
        output = output.unsqueeze(1)
        online_representation = self.online_encoder_network(output)
        online_representation_output = self.output_representation(online_representation)
        online_representation_reshape = online_representation_output.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        return online_representation_reshape

    def get_projection(self, x):
        output_line = self.online_pre_network(x)
        output = output_line.unsqueeze(1)
        online_representation = self.online_encoder_network(output)

        online_representation_output = self.output_representation(online_representation)
        online_representation_permute = online_representation_output.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        B1, T1, D1, C1 = online_representation_permute.shape
        online_representation_reshape = online_representation_permute.reshape((B1, T1 * C1 * D1))
        online_projection = self.online_projector_network(online_representation_reshape)
        return online_projection, (output_line, online_representation_permute)


    def forward(self, x01, x02):
        # 먼저 target network 파라미터부터 따와서 생성
        if self.target_pre_network is None or self.target_projector_network is None:
            self.get_pre_network()
            self.get_target_projector()

        # online network 관련 코드부터 실행 (x01과 x02 모두)
        # input: (batch, frequency, timestep)
        # output: (batch, frequency, timestep)
        online_x01_pre = self.online_pre_network(x01)
        online_x02_pre = self.online_pre_network(x02)
        B1, F1, T1 = online_x01_pre.shape
        B2, F2, T2 = online_x02_pre.shape
        online_representation01_reshape = online_x01_pre.reshape((B1, T1 * F1))
        online_representation02_reshape = online_x02_pre.reshape((B2, T2 * F2))
        online_projection01 = self.online_projector_network(online_representation01_reshape)
        online_projection02 = self.online_projector_network(online_representation02_reshape)
        online_prediction01 = self.online_predictor_network(online_projection01)
        online_prediction02 = self.online_predictor_network(online_projection02)

        with torch.no_grad():
            target_x01_pre = self.target_pre_network(x01)
            target_x02_pre = self.target_pre_network(x02)
            B1, F1, T1 = target_x01_pre.shape
            B2, F2, T2 = target_x02_pre.shape
            target_representation01_reshape = target_x01_pre.reshape((B1, T1 * F1))
            target_representation02_reshape = target_x02_pre.reshape((B2, T2 * F2))
            target_projection01 = self.target_projector_network(target_representation01_reshape)
            target_projection02 = self.target_projector_network(target_representation02_reshape)
        loss01 = self.criterion(online_prediction01, target_projection02.detach())
        loss02 = self.criterion(online_prediction02, target_projection01.detach())
        loss = loss01 + loss02
        online_representation = [(online_x01_pre, online_x02_pre,)]
        target_representation = [(target_x01_pre, target_x02_pre,)]
        return online_representation, target_representation, loss.mean()


if __name__ == '__main__':

    test_model = WaveBYOL(
        config=None,
        pre_input_dims=1,
        pre_hidden_dims=512,
        pre_filter_sizes=[10, 8, 4, 4, 4],
        pre_strides=[5, 4, 2, 2, 2],
        pre_paddings=[2, 2, 2, 2, 1],
        dimension=65536,  # b4,15200 -> 86016
        hidden_size=256,
        projection_size=4096,
    ).cuda()
    print(test_model)
    input_data01 = torch.rand(2, 1, 20480).cuda()
    input_data02 = torch.rand(2, 1, 20480).cuda()
    online_representation_output, target_representation_output, loss = test_model(input_data01, input_data02)
    print(loss)
