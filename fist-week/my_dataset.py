import os
import pandas as pd
from torchvision.io import decode_image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import torch

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),      # 可以先稍微放大一点
    transforms.RandomResizedCrop(224),  # 再随机裁剪到224，效果更好
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),      # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 颜色增强
    transforms.ToTensor(),              # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 标准化
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# target_trans =
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_excel(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]))
        try:
            image = Image.open(img_path).convert("RGB")  # 转换为RGB以确保3通道
        except FileNotFoundError:
            print(f"错误: 找不到文件 {img_path}")
            # 您可以在这里返回一个占位符图像或抛出异常
            return None, None  # 或者更健壮的处理
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_dataset = CustomImageDataset("../dataset/annotefile_train.xlsx", "../dataset/cls_train", transform=train_transform)
val_dataset = CustomImageDataset("../dataset/annotefile_val.xlsx", "../dataset/cls_test", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

# 查看train_dataloader里面的数据长什么样子
for batch, (X, y) in enumerate(train_loader):
    if batch == 0:
        print("这是第一个批次")
        print(f"共用{len(train_loader)}个批次")
        print(f"X的形状为{X.shape}")
        print(f"y的形状为{y}")
        break

# class LeNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=18, kernel_size=3, stride=2, padding=1), # 112
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),      # 56
#
#             nn.Conv2d(in_channels=18, out_channels=48, kernel_size=3, stride=2, padding=1), # 28
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),   # 14
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=9408, out_features=1024),
#             nn.ReLU(),
#             nn.Linear(in_features=1024, out_features=512),
#             nn.ReLU(),
#             nn.Linear(in_features=512, out_features=128),
#             nn.ReLU(),
#             nn.Linear(in_features=128, out_features=2)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         logits = self.classifier(x)
#         return logits


# 添加了batchnorm和dropout层的改进版本
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取部分 (卷积)
        self.features = nn.Sequential(
            # 卷积块 1
            nn.Conv2d(in_channels=3, out_channels=18, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(18),  # <-- 添加 BatchNorm2d
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 卷积块 2
            nn.Conv2d(in_channels=18, out_channels=48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),  # <-- 添加 BatchNorm2d
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 分类器部分 (全连接)
        """
        使用以下代码可以直接得出第一个全连接层的大小应为多少
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224) # (B, C, H, W)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1] # 计算展平后的大小
        
        """

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 全连接块 1

            nn.Linear(in_features=48 * 14 * 14, out_features=1024),
            nn.BatchNorm1d(1024),  # <-- 添加 BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.5),  # <-- 添加 Dropout (0.5是常用值)

            # 全连接块 2
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 全连接块 3
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # (通常在最后一层隐藏层后可以省略Dropout)

            # 输出层 (不加BN和Dropout)
            nn.Linear(in_features=128, out_features=2)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits

model = LeNet()
# state_dict = model.state_dict()
# for k, v in state_dict.items():
#     print(f"网络层名称{k}\t网络权重{v}\t权重形状{v.shape}")
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-3
epochs = 10

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):

    model.train()
    num_batch = len(dataloader)
    train_loss = 0
    for batch, (X,y) in enumerate(dataloader):

        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    # 通常我们所说的“训练损失”和“验证损失”指的是一个周期（Epoch）内，所有批次（batch）损失的平均值
    print(f"训练损失为{train_loss/num_batch:>8f}")



def val_loop(dataloader, model, loss_fn):

    model.eval()
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for (X, y) in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batch
    correct /= size
    print(f"Test Error: \n Accuracy: {100 * correct:>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    val_loop(val_loader, model, loss_fn)

print("Done!")

"""
调用训练好的模型的三个步骤：

保存和加载模型：将训练好的最佳权重保存下来，并在需要时加载。

预处理新图片：新图片必须经过和验证集完全相同的预处理流程。

进行预测：将处理好的图片送入模型，得到预测结果并进行解释。
"""
