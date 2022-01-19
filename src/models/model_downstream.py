import torch.nn as nn
import collections


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
                    ('bn01', nn.BatchNorm1d(hidden_dim)),
                    ('act01', nn.ReLU()),
                    ('linear02', nn.Linear(hidden_dim, output_dim))

                ]
            )
        )

    def forward(self, x):
        output = self.classifier(x)
        return output
