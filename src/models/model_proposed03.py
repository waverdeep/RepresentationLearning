import copy
import os
import torch
import torch.nn as nn
import src.losses.criterion as losses
import src.models.model_proposed02 as model_proposed02
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
        return output

    def forward(self, x01, x02):
        # 먼저 target network 파라미터부터 따와서 생성
        if self.target_pre_network is None \
                or self.target_projector_network is None:
            self.get_pre_network()
            self.get_target_projector()

        # online network 관련 코드부터 실행 (x01과 x02 모두)
        # input: (batch, frequency, timestep)
        # output: (batch, frequency, timestep)
        online_x01 = self.online_pre_network(x01)
        online_x02 = self.online_pre_network(x02)
        B1, D1, T1 = online_x01.shape
        B2, D2, T2 = online_x02.shape
        online_x01_reshape = online_x01.reshape((B1, T1 * D1))
        online_x02_reshape = online_x02.reshape((B2, T2 * D2))
        online_projection01 = self.online_projector_network(online_x01_reshape)
        online_projection02 = self.online_projector_network(online_x02_reshape)
        online_prediction01 = self.online_predictor_network(online_projection01)
        online_prediction02 = self.online_predictor_network(online_projection02)

        with torch.no_grad():
            # input: (batch, frequency, timestep)
            # output: (batch, frequency, timestep)
            target_x01 = self.target_pre_network(x01)
            target_x02 = self.target_pre_network(x02)

            B1, D1, T1 = target_x01.shape
            B2, D2, T2 = target_x02.shape
            target_x01 = target_x01.reshape((B1, T1 * D1))
            target_x02 = target_x02.reshape((B2, T2 * D2))

            # target line은  projection만 시킨다~
            target_projection01 = self.target_projector_network(target_x01)
            target_projection02 = self.target_projector_network(target_x02)

        # 정말 loss 구하는 공식이 맞는지 잘 모르겠지만 일단 해본다
        # detach는 gradient 안딸려오게 복사하는 것~~
        loss01 = self.criterion(online_prediction01, target_projection02.detach())
        loss02 = self.criterion(online_prediction02, target_projection01.detach())
        loss = loss01 + loss02
        return online_x01, loss.mean()


if __name__ == '__main__':
    test_model = WaveBYOL(
        config=None,
        pre_input_dims=1,
        pre_hidden_dims=512,
        pre_filter_sizes=[10, 8, 4, 4, 4],
        pre_strides=[5, 4, 2, 2, 2],
        pre_paddings=[2, 2, 2, 2, 1],
        dimension=65536,
        hidden_size=256,
        projection_size=4096
    ).cuda()
    input_data01 = torch.rand(8, 1, 20480).cuda()
    input_data02 = torch.rand(8, 1, 20480).cuda()
    output, _ = test_model(input_data01, input_data02)
    print(output.size())
    print(_)




