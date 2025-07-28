import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import resnet34
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    # --- 在这里添加关键的 Resize 步骤 ---
    transforms.Resize((224, 224)),

    # 转换为张量 (这一步会自动将像素值从 0-255 缩放到 0-1)
    ToTensor(),

    # (可选) 之后还可以添加标准化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


training_data = datasets.CIFAR10(
    root="second-week/data",
    train=True,
    download=True,
    transform=data_transform
)

test_data = datasets.CIFAR10(
    root="second-week/data",
    train=False,
    download=True,
    transform=data_transform
)



train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)       # 这里面的数据长什么样子？
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# 查看train_dataloader里面的数据长什么样子
# for batch, (X, y) in enumerate(train_dataloader):
#     if batch == 0:
#         print("这是第一个批次")
#         print(f"共用{len(train_dataloader)}个批次")
#         print(f"X的形状为{X.shape}")                      # X的形状为torch.Size([64, 1, 28, 28])
#         print(f"y的形状为{y.shape}")                      # y的形状为torch.Size([64])，64应该是64张图片分别对应的标签
#         break
# 定义的网络，可以考虑懂了训练和验证逻辑之后再加一个batch normalization 和 dropout层
# 线性层 -> 批次归一化 -> 激活函数 -> Dropout

resnet34 = resnet34().to(device)

# 1、设置超参数
learning_rate = 1e-3
epochs = 15
batch_size = 64

# 2、设置损失函数
loss_fn = nn.CrossEntropyLoss()

# 3、设置优化方法
optimizer = torch.optim.SGD(resnet34.parameters(), lr=learning_rate)

# 设置训练循环
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)                    # 查看train_dataloader里面有多少组训练数据
    model.train()                                     # 将模型设置成训练模式
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()                        # 梯度重新设置为0
        pred = model(X)                               # 前向传播
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()


        # 输出设置
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)    # .item() 来从一个单元素张量中获取标准的Python数字，.tolist()将一个包含多个元素的张量转换为标准的Python对象
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")     # 这个f后面的大括号里面的输出语法控制是什么样子的

def evaluate_loop(dataloader, model, loss_fn):
    model.eval()                                                      # 将模型切换到评估模式
    size = len(dataloader.dataset)                                    # 获取测试数据集中的样本总数
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():                                             # 上下文管理器，告诉Pytorch接下来的代码不需要计算梯度，很重要
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)                                           # 这里对每一张图片的预测结果都是Logits (原始分数)，如[1.2, -0.5, 3.8, 0.1, -2.4, 1.5, 0.9, 2.1, -1.8, 0.3]
            test_loss += loss_fn(pred, y).item()                      #
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            """
            首先计算pred.argmax(1) == y，argmax(1)沿着类别维度进行判断，会返回那个值最大的元素的索引为4（举例），
            然后与y比较是否相等返回布尔值，.type(torch.float)将布尔值转化成0和1，
            .sum().item() 求和再取出结果
            """

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {100 * correct:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


best_accuracy = 0.0
# 在循环开始前，先将初始模型的权重作为第一个“最佳”权重
# 这样即使模型性能一直下降，best_state_dict也总是有值的
best_state_dict = resnet34.state_dict()

for t in range(epochs):
    # if t == 0:
    #     best_accuracy = 0   # 这样写不好
    print(f"Epoch {t+1}\n------------------------------")
    train_loop(train_dataloader, resnet34, loss_fn, optimizer)
    current_correct = evaluate_loop(test_dataloader, resnet34, loss_fn)
    if current_correct >= best_accuracy:
        best_accuracy = current_correct
        best_state_dict = resnet34.state_dict()

torch.save(best_state_dict, "best_model.pth")