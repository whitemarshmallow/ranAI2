import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 2)  # 假设输出是两类：正常点和故障点

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 25 * 25)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# 加载数据集
dataset = datasets.ImageFolder('datatransfer/heatmaps', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型和优化器
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(5):  # 训练5轮
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')



# # 加载测试图像
# test_image = 'datatransfer/heatmaps/heatmap_test.png'
# image = datasets.folder.default_loader(test_image)  # 加载测试图像
# image = transform(image).unsqueeze(0)  # 应用相同的变换并添加批量维度
#
# # 预测
# model.eval()
# with torch.no_grad():
#     output = model(image)
#     _, predicted = torch.max(output, 1)
#     if predicted.item() == 1:
#         print("检测到故障点")
#     else:
#         print("未检测到故障点")
