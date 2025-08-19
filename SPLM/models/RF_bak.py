import torch.nn as nn
import math
import torch


class RF_bak(nn.Module):

    def __init__(self, features, num_classes=100, init_weights=True):
        super(RF_bak, self).__init__()
        self.first_layer_in_channel = 1
        self.first_layer_out_channel = 32
        # 添加输入归一化层
        self.input_norm = nn.BatchNorm2d(self.first_layer_in_channel)
        self.first_layer = make_first_layers(self.first_layer_in_channel, self.first_layer_out_channel)
        self.features = features
        self.class_num = num_classes
        self.classifier = nn.AdaptiveAvgPool1d(1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # x = self.input_norm(x)  # 添加归一化
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class RF(nn.Module): # 接收2*100的输入

    def __init__(self, features, num_classes=100, init_weights=True):
        super(RF, self).__init__()
        self.first_layer_in_channel = 1
        self.first_layer_out_channel = 32
        self.first_layer = make_first_layers(self.first_layer_in_channel, self.first_layer_out_channel)
        self.features = features
        self.class_num = num_classes
        self.classifier = nn.AdaptiveAvgPool1d(1)
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Flatten(),
        #     nn.Linear(self.first_layer_out_channel, 256)
        # )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #print("X shape: ",x.shape)
        x = torch.unsqueeze(x, 1)
        x = x.view(x.size(0), 1, 2, -1)
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, in_channels=32):
    layers = []

    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            layers += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
            in_channels = v

    return nn.Sequential(*layers)


def make_first_layers(in_channels=1, out_channel=32):
    layers = []
    conv2d1 = nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d1, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d2, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

    conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d3, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d4, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]

    return nn.Sequential(*layers)


cfg = {
    'N': [128, 128, 'M', 256, 256, 'M', 512]
}

cfg = {
    'N': [128, 'M', 256, 'M', 512]
}

# CNN with 2 dim input
class CNNClassifier(nn.Module):
    def __init__(self, in_channels, seq_len, num_classes, dropout=0.2):
        super(CNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),  # 1D卷积
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 最大池化
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 计算卷积和池化后的序列长度
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * (seq_len // 4), 256),  # 假设经过两次池化，序列长度变为 seq_len / 4
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)  # 输出类别数
        )

    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        x = self.conv_layers(x)  # 通过卷积层
        x = self.flatten(x)  # 展平为 (batch_size, 128 * (seq_len // 4))
        x = self.fc(x)  # 通过全连接层
        return x  # 输出形状: (batch_size, num_classes)


def getRF(num):
    model = RF_bak(make_layers(cfg['N'] + [num]), num_classes=num)
    #model = CNNClassifier(2, 100, num)
    return model

if __name__ == '__main__':
    net = getRF(100)
    print(net)
