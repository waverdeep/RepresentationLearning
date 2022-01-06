import torch.nn as nn
import collections
import torch


class DownstreamClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embedding_size=None):
        super(DownstreamClassification, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(input_dim, hidden_dim)),
                    ('bn01', nn.BatchNorm1d),
                    ('act01', nn.ReLU()),
                    ('linear02', nn.Linear(hidden_dim, output_dim))

                ]
            )
        )

    def forward(self, x):
        pass


class SpeakerClassification(nn.Module):
    def __init__(self, hidden_dim, speaker_num, embedding_size=128):
        super(SpeakerClassification, self).__init__()
        self.speaker_num = speaker_num
        self.hidden_dim = hidden_dim
        self.adaptive_avg_pool1d = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(
            collections.OrderedDict(
                [
                    ('embedding_linear01', nn.Linear(hidden_dim, embedding_size)),
                ]
            )
        )
        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('embedding_bn01', nn.BatchNorm1d(embedding_size)),
                    ('embedding_relu01', nn.ReLU()),
                    ('classifier_linear01', nn.Linear(embedding_size, speaker_num))
                ]
            )
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.adaptive_avg_pool1d(x)
        x = x.permute(0, 2, 1).reshape(-1, self.hidden_dim)
        embedding_vector = self.embedding(x)
        output = self.classifier(embedding_vector)
        return embedding_vector, self.softmax(output)


class PhonemeClassification(nn.Module):
    def __init__(self, hidden_dim, phoneme_num):
        super(PhonemeClassification, self).__init__()
        self.phoneme_num = phoneme_num
        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(hidden_dim, phoneme_num))
                ]
            )
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self):
        pass


if __name__ == '__main__':
    model = SpeakerClassification(256, 251).cuda()
    joy_data = torch.rand(8, 128, 256).cuda()
    embed, output = model(joy_data)
    print(embed.size())
    print(output.size())
