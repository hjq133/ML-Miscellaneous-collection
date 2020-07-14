# 利用Pytorch解决XOR问题
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
title = "name and number(4)"

data = np.array([[1, 0, 1], [0, 1, 1],
                 [1, 1, 0], [0, 0, 0]], dtype='float32')
x = data[:, :2]
y = data[:, 2]


class XOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)  # 隐藏层 4个神经元
        self.fc2 = nn.Linear(4, 4)  # 隐藏层 4个神经元
        self.fc3 = nn.Linear(4, 1)  # 输出层 1个神经元
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.sigmoid(self.fc1(x))
        h2 = self.sigmoid(self.fc2(h1))
        h3 = self.sigmoid(self.fc3(h2))
        return h3


x = torch.Tensor(x.reshape(-1, 2))
y = torch.Tensor(y.reshape(-1, 1))
net = XOR()

# 定义loss function
criterion = nn.BCELoss()  # MSE
# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)  # SGD
# 训练
for epoch in range(10000):
    optimizer.zero_grad()  # 清零梯度缓存区
    out = net(x)
    loss = criterion(out, y)
    print(loss.item())
    loss.backward()
    optimizer.step()  # 更新

# 可视化输出
x = np.linspace(start=-0.5, stop=1.5, num=100)
y = np.linspace(start=-0.5, stop=1.5, num=100)
data = []
for i in range(len(x)):
    for j in range(len(y)):
        data.append([x[i], y[j]])
data = torch.Tensor(data)
output = net(data)
output = output.detach().numpy()
output[output > 0.5] = 1
output[output < 0.5] = 0

fig, ax = plt.subplots()
for i in range(len(data)):
    x = data[i][0].item()
    y = data[i][1].item()
    if output[i] == 1:
        color = 'red'
    else:
        color = 'blue'
    ax.scatter(x, y, color=color)
ax.legend()
ax.set_title(title, fontsize=12, color='black')
ax.grid(True)
ax.set_aspect(1)
plt.show()
