import torch.nn as nn
import collections


class SpeakerClassification(nn.Module):
    def __init__(self, hidden_dim, speaker_num):
        super(SpeakerClassification, self).__init__()
        self.speaker_num = speaker_num
        self.adaptive_avg_pool1d = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear01', nn.Linear(hidden_dim, speaker_num))
                ]
            )
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.adaptive_avg_pool1d(x)
        x = x.permute(0, 2, 1).reshape(-1, 256)
        output = self.classifier(x)
        return self.softmax(output)
