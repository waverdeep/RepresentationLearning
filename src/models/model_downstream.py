import torch.nn as nn
import collections


class SpeakerClassification(nn.Module):
    def __init__(self, hidden_dim, speaker_num):
        super(SpeakerClassification, self).__init__()
        self.speaker_num = speaker_num

        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(hidden_dim, speaker_num))
                    # ('linear01', nn.Linear(256, 512)),
                    # ('bn01', nn.BatchNorm1d(512)),
                    # ('relu01', nn.ReLU()),
                    # ('linear02', nn.Linear(512, self.speaker_num)),
                ]
            )
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        output = self.classifier(x)
        return self.softmax(output)
