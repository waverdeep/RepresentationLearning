import torch
import torch.nn as nn
import collections
import numpy as np
import json
import src.losses.loss_baseline as criterion


class CPCType01(nn.Module):
    def __init__(self, timestep, audio_window):
        super(CPCType01, self).__init__()
        self.timestep = timestep
        self.audio_window = audio_window
        # We use five convolutional layers with strides [5, 4, 2, 2, 2],
        # filter-sizes [10, 8, 4, 4, 4] and
        # 512 hidden units with ReLU activations

        # The total downsampling factor of the network is 160 so that there is a feature vector for every 10ms of speech
        self.encoder = nn.Sequential(
            collections.OrderedDict(
                [
                    ('encoder_conv01', nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=True)),
                    ('encoder_bn01', nn.BatchNorm1d(512)),
                    ('encoder_relu01', nn.ReLU(inplace=True)),

                    ('encoder_conv02', nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=3, bias=True)),
                    ('encoder_bn02', nn.BatchNorm1d(512)),
                    ('encoder_relu02', nn.ReLU(inplace=True)),

                    ('encoder_conv03', nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=3, bias=True)),
                    ('encoder_bn03', nn.BatchNorm1d(512)),
                    ('encoder_relu03', nn.ReLU(inplace=True)),

                    ('encoder_conv04', nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=3, bias=True)),
                    ('encoder_bn04', nn.BatchNorm1d(512)),
                    ('encoder_relu04', nn.ReLU(inplace=True)),

                    ('encoder_conv05', nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=3, bias=True)),
                    ('encoder_bn05', nn.BatchNorm1d(512)),
                    ('encoder_relu05', nn.ReLU(inplace=True)),
                ]
            )
        )
        # We then use a GRU RNN [17] for the autoregressive part of the model, gar with 256 dimensional hidden state.
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)

        # The output of the GRU at every timestep is used as the context c from which we predict 12 timesteps
        # in the future using the contrastive loss.
        self.Wk = nn.ModuleList([nn.Linear(256, 512) for i in range(self.timestep)])
        self.softmax = nn.Softmax(1)
        self.log_softmax = nn.LogSoftmax(1)

        # weight initialize -> 근데 이거 왜하는지는 잘 모르겠음
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def forward(self, x, hidden):
        # 배치수 추출
        batch = x.size()[0]

        # randomly pick time stamps
        t_samples = torch.randint(int(self.audio_window/160-self.timestep), size=(1, )).long()

        # input sequence: batch*channel*length, N*C*L, 8*1*20480
        z = self.encoder(x)
        # output sequence: batch*channel*length,N*C*L, x*512*b, 8*512*128
        # reshape to batch*length*channel for GRU: N*L*C, x*b*51, 8*125*512
        z = z.transpose(1, 2)

        # average over timestep and batch
        nce = 0
        # timestep*batch*512, 12*8*512 // 이 부분이 zt+k 부분에 해당하는 것 같음
        encode_samples = torch.empty((self.timestep, batch, 512)).float()
        for i in np.arange(1, self.timestep+1):
            # z_tk, 8*512
            encode_samples[i-1] = z[:, t_samples+i, :].view(batch, 512)
        # 8*100*512 // 이 부분이 context_vector를 뽑아내기 위해 autoregressive model을 통과하는 값들을 의미하는 것 같음
        # 나중에 negative 로 사용되지 않나?
        forward_seq = z[:, :t_samples + 1, :]

        # output size 8*100*256
        output, hidden = self.gru(forward_seq, hidden)
        # 8*256 // context vector는 가장 마지막에 있는 ct
        c_t = output[:, t_samples, :].view(batch, 256)

        # 12*8*512 // 예측값을 담을 공간, 위에서만든 encode_sample과 동일한 크기로 텐서 생성
        pred = torch.empty((self.timestep, batch, 512)).float()

        # 0부터 timestep만큼 반복하며 pred 텐서를 c_t의 linear를 거친 형태로 채움
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            # Wk*c_t, size 8*512
            pred[i] = linear(c_t)

        # 0부터 timestep만큼 반복
        for i in np.arange(0, self.timestep):
            # encode_sample -> 8, 512
            # pred -> 8, 512 -> transpose -> 512, 8
            # matrix multiplication 8*8
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            # 8*8 을 softmax에 넣어서 0-1 사이 으로 바꿔주고 (합계가 1이 되게끔),
            # 정답의 개수를 세어서 (eq) correct 변수에 담는다
            # total의 값을 로그소프트맥스 취한 후 대각으로 배치한후 nce 변수에 대입함
            # nce를 -1*배치*timestep으로 니눈후 대입하고
            # 1*correct.item()을 배치로 나눈 값을 accuracy로 대입함
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.log_softmax(total)))  # nce is a tensor
        nce /= -1. * batch * self.timestep
        accuracy = 1. * correct.item() / batch

        return accuracy, nce, hidden


class CPCType02(nn.Module):
    def __init__(self, args, g_enc_input, g_enc_hidden, g_ar_hidden, filter_sizes, strides, paddings):
        super(CPCType02, self).__init__()
        self.encoder = Encoder(input_dim=g_enc_input,
                               hidden_dim=g_enc_hidden,
                               filter_sizes=filter_sizes,
                               strides=strides,
                               paddings=paddings,
                               )
        self.autoregressive = AutoRegressive(input_dim=g_enc_hidden,
                                             hidden_dim=g_ar_hidden,
                                             )
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

        self.encoder= nn.Sequential()
        for index, (stride, filter_size, padding) in enumerate(zip(strides, filter_sizes, paddings)):
            self.encoder.add_module(
                "encoder_layer_{}".format(index),
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, filter_size, stride=stride, padding=padding),
                    nn.ReLU(),
                )
            )
            input_dim = hidden_dim

    def forward(self, x):
        return self.encoder(x)


class AutoRegressive(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        super(AutoRegressive, self).__init__()
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


def init_hidden(batch_size, use_gpu=True):
    if use_gpu:
        return torch.zeros(1, batch_size, 256).cuda()
    else:
        return torch.zeros(1, batch_size, 256)


if __name__ == '__main__':
    # CPC_model = CPCType01(12, 20480).cuda()
    # print(CPC_model.modules())
    # joy_data = torch.rand(8, 1, 20480).cuda()
    # accaracy, nce, hidden = CPC_model(joy_data, init_hidden(8))
    # print("acc: ", accaracy)
    # print("nce: ", nce)
    # print("hidden: ", hidden)
    with open('../../config/config_type02_direct_train03.json', 'r') as configuration:
        config = json.load(configuration)
    model = CPCType02(args=config,
                      g_enc_input=config['model']['g_enc_input'],
                      g_enc_hidden=config['model']['g_enc_hidden'],
                      g_ar_hidden=config['model']['g_ar_hidden'],
                      filter_sizes=config['model']['filter_sizes'],
                      strides=config['model']['strides'],
                      paddings=config['model']['paddings']).cuda()
    joy_data = torch.rand(8, 1, 20480).cuda()
    _loss, _accuracy, _z, _c = model(joy_data)
    print(_accuracy)




