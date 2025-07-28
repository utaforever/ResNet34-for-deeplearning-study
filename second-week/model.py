from torch import nn
import torch
import torch.nn.functional as F

#我自己写的
# class Basic_Block(nn.Module):
#     def __init__(self, in_channels, out_channels, downsample=False):
#         super().__init__()
#         self.downsample = downsample
#         if downsample:
#             # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
#             # self.bn1 = nn.BatchNorm2d(out_channels)
#             # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#             # self.bn2 = nn.BatchNorm2d(out_channels)
#             self.basic_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
#                                         nn.BatchNorm2d(out_channels), nn.ReLU(),
#                                         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                                         nn.BatchNorm2d(out_channels))
#             self.identity_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
#                                            nn.BatchNorm2d(out_channels))
#         else:
#             self.basic_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
#                                         nn.BatchNorm2d(out_channels),nn.ReLU(),
#                                         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                                         nn.BatchNorm2d(out_channels))
#
#
#     def forward(self, X):
#         if self.downsample:
#             return F.relu(self.basic_block(X) + self.identity_block(X))
#         else:
#             return F.relu(self.basic_block(X) + X)


# 标准的代码
class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # --- 主路径 (Main Path) ---
        # 注意这里不包含最后的ReLU
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 第一个ReLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # --- 捷径 (Shortcut Path) ---
        self.shortcut = nn.Sequential()  # 默认是一个空的“直连”通道
        # 如果需要下采样 (stride不为1) 或者输入输出通道数不同
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 使用1x1卷积进行维度匹配，没有ReLU
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # 最后的激活函数单独定义
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 主路径的输出
        out_main = self.main_path(x)

        # 捷径的输出
        out_shortcut = self.shortcut(x)

        # 先相加
        out = out_main + out_shortcut

        # 后激活
        out = self.final_relu(out)

        return out


# test_inputs = torch.randn(4, 3, 224, 224)
# model = Basic_Block(3, 6, 2)
# print(model)
# print(model(test_inputs).shape)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # def _make_layer(in_channels, out_channels, num_layers, stride=1):
        #     layers = []
        #     for i in range(num_layers):
        #         if i == 0 and stride != 1:
        #             layers.append(Basic_Block(in_channels, out_channels, stride=stride))
        #             continue
        #         layers.append(Basic_Block(out_channels, out_channels))
        #     return nn.Sequential(*layers)

        # 一种更为简洁的写法
        def _make_layer(in_channels, out_channels, num_blocks, stride=1):


            layers = []

            # 第一个块，它可能需要处理下采样和通道数变化
            layers.append(Basic_Block(in_channels, out_channels, stride=stride))

            # 剩下的 num_blocks - 1 个块都是恒等块
            for _ in range(1, num_blocks):
                layers.append(Basic_Block(out_channels, out_channels, stride=1))

            return nn.Sequential(*layers)

        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                    )
        self.layer2 = _make_layer(64, 64, num_blocks=3)

        self.layer3 = _make_layer(64, 128, num_blocks=4, stride=2)

        self.layer4 = _make_layer(128, 256, num_blocks=6, stride=2)

        self.layer5 = _make_layer(256, 512, num_blocks=3, stride=2)

        # self.layer6 = nn.Sequential(nn.AvgPool2d(7), nn.Flatten(), nn.Linear(512, 10))

        self.layer6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))

    def forward(self, X):
        out1 = self.layer1(X)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        return out6


def resnet34():

    return ResNet()




