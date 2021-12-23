import sys
import torch
import torch.nn as nn
import src.models.model_baseline as model_baseline
import src.losses.criterion as losses
import src.utils.interface_tensor_manipulation as tensor_manipulation
import json


class Decoder(nn.Module):
    def __init__(self, input_dims=[512, 256, 128, 64, 32, 16], hidden_dims=[256, 128, 64, 32, 16, 1],
                 filter_sizes=[5, 2, 2, 2, 2, 2], strides=[5, 2, 2, 2, 2, 2]):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential()
        for index, (input_dim, hidden_dim, filter_size, stride) in enumerate(zip(
                input_dims, hidden_dims, filter_sizes, strides)):
            self.decoder.add_module(
                "decoder_layer_{}".format(index),
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels=input_dim, out_channels=hidden_dim,
                                       kernel_size=filter_size, stride=stride),
                    nn.ReLU(inplace=False),
                )
            )

    def forward(self, x):
        output = self.decoder(x)
        return output


class GenerativeCPCModel(model_baseline.CPCModel):
    def __init__(self, args, g_enc_input, g_enc_hidden, g_ar_hidden, filter_sizes, strides, paddings):
        super().__init__(args, g_enc_input, g_enc_hidden, g_ar_hidden, filter_sizes, strides, paddings)
        self.decoder = Decoder()
        self.decoder_loss = losses.set_criterion("L1Loss")

    def get_latent_representations(self, x):
        z = self.encoder(x)
        permuted_z = z.permute(0, 2, 1)
        c = self.autoregressive(permuted_z)
        return z, permuted_z, c

    def alteration(self, z):
        alterated_z = tensor_manipulation.random_alteration_tensor_1d(z)
        return alterated_z

    def generative_network(self, z):
        alterated_z = self.alteration(z)
        generated_x = self.decoder(alterated_z)
        return generated_x

    def forward(self, x):
        z, _, _ = self.get_latent_representations(x)
        _, permuted_z, c = self.get_latent_representations(x)
        generated_x = self.generative_network(z)
        decoder_loss_output = self.decoder_loss(generated_x, x)
        loss, accuracy = self.loss.get(x, permuted_z, c)
        total_loss = loss + decoder_loss_output
        return total_loss, accuracy, z, c


if __name__ == '__main__':
    with open('../../config/config_pretext-GCPC-kspon-training01-batch64.json', 'r') as configuration:
        config = json.load(configuration)
    model = GenerativeCPCModel(args=config,
                     g_enc_input=1,
                     g_enc_hidden=512,
                     g_ar_hidden=256,
                     filter_sizes=config['filter_sizes'],
                     strides=config['strides'],
                     paddings=config['paddings']).cuda()
    for idx, layer in enumerate(model.modules()):
        print(layer)
    test_loss, test_accuracy, test_z, test_c = model(torch.rand(32, 1, 20480).cuda())
    print(test_loss.size())
    print(test_loss)