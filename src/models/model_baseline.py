import torch
import torch.nn as nn
import json
import src.losses.loss_baseline as criterion


class CPCModel(nn.Module):
    def __init__(self, args, g_enc_input, g_enc_hidden, g_ar_hidden, filter_sizes, strides, paddings):
        super(CPCModel, self).__init__()
        self.encoder = Encoder(input_dim=g_enc_input,
                               hidden_dim=g_enc_hidden,
                               filter_sizes=filter_sizes,
                               strides=strides,
                               paddings=paddings, )
        self.autoregressive = AutoRegressive(input_dim=g_enc_hidden,
                                             hidden_dim=g_ar_hidden,)
        self.loss = criterion.InfoNCE(args=args, gar_hidden=g_ar_hidden, genc_hidden=g_enc_hidden)

    # pretext model에서 latent vector들을 추출하는 함수 작성함 (latent z, ct)
    def get_latent_representations(self, x):
        z = self.encoder(x)
        # tensor.permute는 차원을 변경하는 함수이다. 현재는 L과 C 차원을 변경한다.
        z = z.permute(0, 2, 1)
        # autoregressive모델을 쳐서 만들어진 vector를 추출한다 -> context를 뜻
        # 여기서 모든 timestep에 대한 c를 구해야하 하는지 좀 의문임
        c = self.autoregressive(z)
        return z, c

    def get_latent_size(self, input_size):
        x = torch.zeros(input_size).cuda()
        z, c = self.get_latent_representations(x)
        return c.size(2), c.size(1)

    def forward(self, x):
        z, c = self.get_latent_representations(x)
        # loss 확인 필요
        loss, accuracy = self.loss.get(x, z, c)
        return loss, accuracy, z, c


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, strides, filter_sizes, paddings):
        super(Encoder, self).__init__()
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
                    nn.ReLU(inplace=False),
                )
            )
            input_dim = hidden_dim

    def forward(self, x):
        return self.encoder(x)


class AutoRegressive(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoRegressive, self).__init__()
        self.hidden_dim = hidden_dim
        self.autoregressive = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
        )

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.get_device())
        self.autoregressive.flatten_parameters()
        output, _ = self.autoregressive(x, h0)
        return output


if __name__ == '__main__':
    with open('../../config/config_pretext-CPC(baseline)-kspon-training01-batch64.json', 'r') as configuration:
        config = json.load(configuration)
    model = CPCModel(args=config,
                     g_enc_input=1,
                     g_enc_hidden=512,
                     g_ar_hidden=256,
                     filter_sizes=config['filter_sizes'],
                     strides=config['strides'],
                     paddings=config['paddings']).cuda()
    joy_data = torch.rand(8, 1, 20480).cuda()
    _loss, _accuracy, _z, _c = model(joy_data)
    print(_loss.size())
    print(_accuracy)
    print(_z.size())
    print(_c.size())



