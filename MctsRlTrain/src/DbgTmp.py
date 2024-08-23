'''
DataSet
'''

import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models

# 自定义数据集类
class CustomDataset(Dataset):   # 它继承自torch.utils.data.Dataset，并实现其中的两个方法：__len__和__getitem__
    def __init__(self, csv_file, img_folder, transform=None):
        self.labels_df = pd.read_csv(csv_file)  # self.labels_df是一个DataFrame对象，其中包含图片名和对应的目标标签
        self.img_folder = img_folder
        self.transform = transform  # 这是一个可选参数，用于在数据加载时对图像进行预处理（如缩放、裁剪、标准化等）

    def __len__(self):  # 返回数据集中样本的数量
        return len(self.labels_df)

    def __getitem__(self, idx): # 它接受一个索引idx，并返回对应的图像和标签
        img_name = os.path.join(self.img_folder, self.labels_df.iloc[idx, 0])   # 获取DataFrame中第idx行、第0列的图片名
        image = Image.open(img_name).convert('RGB') # 使用Pillow库的Image.open打开图片，并将其转换为RGB模式
        label = self.labels_df.iloc[idx, 1:].values.astype('float') # 获取目标标签。self.labels_df.iloc[idx, 1:]获取第idx行，第1列及之后的所有列（这些列包含目标标签）

        if self.transform:  # self.transform通常是一个torchvision.transforms.Compose对象，包含多个图像预处理步骤
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float)

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 实例化数据集和数据加载器
dataset = CustomDataset(csv_file=r'D:\DataSet\TrainDataSet\label.csv', img_folder=r'D:\DataSet\TrainDataSet\images', transform=transform)
# batch_size=32指定每个批次包含32个样本
# shuffle=True表示每个epoch开始时将数据集打乱，以增加训练的随机性。
# drop_last=True表示如果最后一个批次样本数不足以形成完整批次，则丢弃该批次。
# num_workers=4指定使用4个子进程来加载数据，以提高数据加载速度。
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4)  # 批量加载数据集中的数据

# 验证数据加载器是否工作
for images, labels in train_loader:
    print(images.shape, labels.shape)
    break

# 定义ResNet模型
class ResNetActionModel(nn.Module):
    def __init__(self):
        super(ResNetActionModel, self).__init__()
        self.resnet = models.resnet34(pretrained=True)  # 使用预训练的ResNet34
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 6)  # 修改最后的全连接层

    def forward(self, x):
        return self.resnet(x)

# 初始化模型、损失函数和优化器
model = ResNetActionModel()
criterion = nn.MSELoss()  # 均方误差损失函数,通常用于回归任务，衡量模型预测值与真实值之间的差异
optimizer = optim.Adam(model.parameters(), lr=0.001)    # model.parameters()返回模型中的所有可训练参数

# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()   # 将模型设置为训练模式, 会启用诸如Dropout和BatchNorm等训练时特有的功能
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:    # train_loader是一个数据加载器，返回batch_size=32个数据
            optimizer.zero_grad()   # 在每次迭代前清除之前的梯度
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward() # 反向传播损失
            optimizer.step()    # 更新模型参数
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 调用训练函数
train(model, train_loader, criterion, optimizer)
