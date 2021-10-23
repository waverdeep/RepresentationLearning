from src.models.model_baseline import *


class AutoRegressiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoRegressiveLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.autoregressive = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.get_device())
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.get_device())
        self.autoregressive.flatten_parameters()
        output, _ = self.autoregressive(x, (h0, c0))
        return output


class EncoderDilatedCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, strides, filter_sizes, paddings):
        super(EncoderDilatedCNN, self).__init__()
        assert(
                len(strides) == len(filter_sizes) == len(paddings)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.encoder = nn.Sequential()
        for index, (stride, filter_size, padding) in enumerate(zip(strides, filter_sizes, paddings)):
            self.encoder.add_module(
                "encoder_layer_{}".format(index),
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, filter_size, stride=stride, padding=padding, dilation=2),
                    nn.ReLU(),
                )
            )
            input_dim = hidden_dim

    def forward(self, x):
        return self.encoder(x)


class CPCType02LSTM(CPCType02):
    def __init__(self, args, g_enc_input, g_enc_hidden, g_ar_hidden, filter_sizes, strides, paddings):
        super(CPCType02, self).__init__()
        CPCType02.__init__(self, args, g_enc_input, g_enc_hidden, g_ar_hidden, filter_sizes, strides, paddings)
        self.autoregressive = AutoRegressiveLSTM(input_dim=g_enc_hidden,
                                                 hidden_dim=g_ar_hidden)


class CPCType02Dilation(CPCType02):
    def __init__(self, args, g_enc_input, g_enc_hidden, g_ar_hidden, filter_sizes, strides, paddings):
        super(CPCType02, self).__init__()
        CPCType02.__init__(self, args, g_enc_input, g_enc_hidden, g_ar_hidden, filter_sizes, strides, paddings)
        self.encoder = EncoderDilatedCNN(input_dim=g_enc_input,
                                         hidden_dim=g_enc_hidden,
                                         filter_sizes=filter_sizes,
                                         strides=strides,
                                         paddings=paddings, )


if __name__ == '__main__':
    with open('../../config/config_type02_direct_train03.json', 'r') as configuration:
        config = json.load(configuration)
    print(config)
    model = CPCType02LSTM(args=config,
                          g_enc_input=1,
                          g_enc_hidden=512,
                          g_ar_hidden=256,
                          filter_sizes=config['model']['filter_sizes'],
                          strides=config['model']['strides'],
                          paddings=config['model']['paddings']).cuda()
    joy_data = torch.rand(8, 1, 32000).cuda()
    _loss, _accuracy, _z, _c = model(joy_data)
    print(_accuracy)

    model = CPCType02Dilation(args=config,
                              g_enc_input=1,
                              g_enc_hidden=512,
                              g_ar_hidden=256,
                              filter_sizes=config['model']['filter_sizes'],
                              strides=config['model']['strides'],
                              paddings=config['model']['paddings']).cuda()
    joy_data = torch.rand(8, 1, 32000).cuda()
    _loss, _accuracy, _z, _c = model(joy_data)
    print(_accuracy)