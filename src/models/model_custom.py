import torch
import torch.nn as nn
import src.models.model_resnet as resnet
import torchvision.models as models


class ResNetCombine(nn.Module):
    def __init__(self, config, embedding_size, num_classes, backbone='resnet18', pretrained=False):
        super(ResNetCombine, self).__init__()

        if backbone == 'resnet18':
            self.pretrained_model = resnet.resnet18(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.pretrained_model = resnet.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.pretrained_model = resnet.resnet152(pretrained=pretrained)
        elif backbone == 'resnet18':
            self.pretrained_model = resnet.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.pretrained_model = resnet.resnet34(pretrained=pretrained)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0 = nn.Linear(128, embedding_size)
        self.bn0 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        self.last = nn.Linear(embedding_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pretrained_model.conv1(x)
        x = self.pretrained_model.bn1(x)
        x = self.pretrained_model.relu(x)
        x = self.pretrained_model.layer1(x)
        x = self.pretrained_model.layer2(x)
        x = self.pretrained_model.layer3(x)
        x = self.pretrained_model.layer4(x)
        out = self.avg_pool2d(x)
        out = torch.squeeze(out)
        out = out.view(x.size(0), -1)
        spk_embedding = self.fc0(out)
        out = self.relu(self.bn0(spk_embedding))  # [batch, n_embed]
        out = self.last(out)
        # print(spk_embedding.size())
        # print(out.size())
        return spk_embedding, self.softmax(out)


class VGGCombine(nn.Module):
    def __init__(self, config, num_classes, embedding_size):
        super(VGGCombine, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.conv01 = nn.Conv2d(1, 3, 3)
        self.vgg16 = models.vgg16(pretrained=True).features
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0 = nn.Linear(512, embedding_size)
        self.bn0 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        self.last = nn.Linear(embedding_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv01(x)
        out = self.vgg16(out)
        out = self.avg_pool2d(out)
        out = torch.squeeze(out)
        out = out.view(x.size(0), -1)
        spk_embedding = self.fc0(out)
        out = self.relu(self.bn0(spk_embedding))  # [batch, n_embed]
        out = self.last(out)
        print(spk_embedding.size())
        print(out.size())
        return spk_embedding, self.softmax(out)


class MobileNetV2Combine(nn.Module):
    def __init__(self, config, num_classes, embedding_size):
        super(MobileNetV2Combine, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.conv01 = nn.Conv2d(1, 3, 3)
        self.mobile = models.mobilenet_v2(pretrained=True).features
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0 = nn.Linear(1280, embedding_size)
        self.bn0 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        self.last = nn.Linear(embedding_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv01(x)
        out = self.mobile(out)
        out = self.avg_pool2d(out)
        out = torch.squeeze(out)
        out = out.view(x.size(0), -1)
        spk_embedding = self.fc0(out)
        out = self.relu(self.bn0(spk_embedding))  # [batch, n_embed]
        out = self.last(out)
        print(spk_embedding.size())
        print(out.size())
        return spk_embedding, self.softmax(out)


class SqueezeNetCombine(nn.Module):
    def __init__(self, config, num_classes, embedding_size):
        super(SqueezeNetCombine, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.conv01 = nn.Conv2d(1, 3, 3)
        self.squeeze = models.squeezenet1_0(pretrained=True).features
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0 = nn.Linear(512, embedding_size)
        self.bn0 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        self.last = nn.Linear(embedding_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv01(x)
        out = self.squeeze(out)
        out = self.avg_pool2d(out)
        out = torch.squeeze(out)
        out = out.view(x.size(0), -1)
        spk_embedding = self.fc0(out)
        out = self.relu(self.bn0(spk_embedding))  # [batch, n_embed]
        out = self.last(out)
        print(spk_embedding.size())
        print(out.size())
        return spk_embedding, self.softmax(out)


if __name__ == '__main__':
    # model = ResNetCombine(None, 128, 251, backbone='resnet18', pretrained=False)
    model = SqueezeNetCombine(None, 128, 251).cuda()
    input = torch.zeros((8, 1, 256, 41)).cuda()
    model(input)


