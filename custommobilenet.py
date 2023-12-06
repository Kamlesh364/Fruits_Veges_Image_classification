import torch
import torch.nn as nn
from urllib.request import urlretrieve
from os.path import exists
from os import mkdir

class Conv2dNormActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, bias=False):
        super(Conv2dNormActivation, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(Conv2dNormActivation(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1))

        layers.extend([
            Conv2dNormActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
IMAGENET1K_V1 = {"url":"https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                "meta":{
                        "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
                        "_metrics": {
                            "ImageNet-1K": {
                                "acc@1": 71.878,
                                "acc@5": 90.286,
                            }
                        },
                        "_ops": 0.301,
                        "_file_size": 13.555,
                        "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
                    },
            }

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()

        inverted_residual_setting = [
            # t: expansion factor, c: output channels, n: number of repetitions, s: stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(32 * width_mult)  # default width_mult = 1.0
        self.features = [Conv2dNormActivation(3, input_channel, kernel_size=3, stride=2, padding=1, groups=1)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Last Conv2d layer
        self.features.append(Conv2dNormActivation(input_channel, 1280, kernel_size=1, stride=1, padding=0, groups=1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x

def mobilenet_v2(weights=None, num_classes=1000):
    model = MobileNetV2(num_classes=num_classes)
    if weights is None:
        if not exists('fruitData/checkpoints/'):
            mkdir('fruitData/checkpoints/')
        urlretrieve(IMAGENET1K_V1['url'], 'fruitData/checkpoints/mobilenet_v2.pth')
        model.load_state_dict(torch.load('fruitData/checkpoints/mobilenet_v2.pth'))
    else:
        model.load_state_dict(torch.load(weights))
    return model