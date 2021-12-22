import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,input_dims=[256, 128, 64, 32, 16, 8], hidden_dims=[128, 64, 32, 16, 8, 1],
                 filter_sizes=[5, 2, 2, 2, 2, 2], strides=[5, 2, 2, 2, 2, 2]):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential()
        for index, (input_dim, hidden_dim, filter_size, stride) in enumerate(zip(input_dims, hidden_dims, filter_sizes, strides)):
            self.decoder.add_module(
                "decoder_layer_{}".format(index),
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels=input_dim, out_channels=hidden_dim,
                                       kernel_size=filter_size, stride=stride),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        return self.decoder(x)


if __name__ == '__main__':
    model = Decoder()
    input = torch.zeros((8, 1, 256, 128))
    output = model(input)
    print(output.size())
