import torch
from torch import nn
import torch.nn.functional as F

'''
nn.ReLU 是一个继承自 nn.Module 的类。这意味着它本身被设计成神经网络的一个层，就像 nn.Linear（全连接层）或 nn.Conv2d（卷积层）一样。

用法: 你必须先在你的网络 __init__ 方法中实例化它，然后再在 forward 方法中调用。

适用场景: 当你使用 nn.Sequential 容器搭建网络时，或者想让网络结构在 __init__ 中一目了然时，通常会使用它。
'''

'''
F.relu 是定义在 torch.nn.functional 模块中的一个普通函数（通常我们 import torch.nn.functional as F 来简化调用）。

用法: 你不需要实例化它，直接在 forward 方法中像调用任何其他函数一样使用它，把张量作为参数传进去。

适用场景: 当你自定义网络类（继承自 nn.Module）时，在 forward 方法中使用它会显得更简洁、更灵活。
'''

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6, 10)
        self.linear2 = nn.Linear(10, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(self.flatten(x))))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(SimpleNet(), nn.ReLU(), nn.Linear(3, 10), nn.ReLU())
        self.linear3 = nn.Linear(10, 3)

    def forward(self, x):
        return self.linear3(self.net(x))

#前两个代码还可以简化成一个代码：
'''
class BetterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 原 SimpleNet 的层
            nn.Flatten(),
            nn.Linear(6, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
            
            # 原 Net 中的后续层
            nn.ReLU(),
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        return self.network(x)
'''
x = torch.randn(2, 3, 2)
net = SimpleNet()
print(net)
net1 = Net()
print(f"Net: {net1}")
print(net(x))
print(f"Net的输出结果为{net1(x)}")

#访问网络的参数
state_dict = net1.state_dict()
for k, v in state_dict.items():
    print(f"网络层名称{k}\t网络权重{v}\t权重形状{v.shape}")


parameters = net1.named_parameters()     # 这个方法会返回一个包含 (参数名称, 参数本身) 的元组的迭代器
for name, param in parameters:
    print(f"网络名称:{name}\t网络权重：{param}\t参数形状：{param.shape}")