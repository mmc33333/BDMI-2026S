# Mi0324学习笔记

## Deep Learning

### Multi-Layer Neural Networks

#### Structure

一个多层神经网络由输入层、若干隐藏层和输出层组成：

```
Input layer → Hidden layer 1 → ... → Hidden layer n → Output layer → Softmax
```

用张量表示三层网络（3输入，3隐藏，2输出）：
- $X, W_1, H, W_2, N, Y$ 均为张量
- $H = W_1 \cdot X$，$N = W_2 \cdot H$，$Y = \text{Softmax}(N)$

```python
import numpy as np

X = np.array([22, 35, 86])
W1 = np.random.randn(3, 3)
W2 = np.random.randn(2, 3)

H = W1 @ X
N = W2 @ H

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

Y = softmax(N)
```

#### Softmax & Logit

Softmax 将 logits 转化为概率分布向量（所有分量为正且和为1）：

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Logit：将 $(0,1)$ 区间映射到 $(-\infty, +\infty)$

$$\text{logit}(p) = \log\frac{p}{1-p}$$

---

## Automatic Differentiation（自动微分）

深度学习的核心：**计算损失函数对权重的偏导数**

### 三种求导方式

#### 1. Numerical Differentiation 数值微分

根据导数定义近似计算：

$$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$

缺点：存在舍入误差，计算效率低

#### 2. Symbolic Differentiation 符号微分

从表达式出发，直接得到导数表达式。

缺点：函数越复杂表达式指数级膨胀，需要保存大量中间变量

#### 3. Automatic Differentiation 自动微分（核心）

结合数值微分精度和符号微分优势，沿计算图传播导数。

- **前向模式（Forward Mode）**：计算所有输出对某个输入的微分，一次前向过程得到 Jacobian 矩阵的一列。

- **反向模式（Reverse Mode）**：从输出到输入反向传播，一次反向过程得到所有输入的梯度（即 Backpropagation）。

  伴随变量：$\bar{v} = \frac{\partial y}{\partial v}$，输入梯度 = 伴随变量 × 局部梯度

> 深度学习中一般采用反向模式，因为输出维度（损失）远小于输入维度（权重数量）。

---

## PyTorch: Autograd 自动微分

```python
import torch

x = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
y = x.sum()       # y = 4
z = y ** 2        # z = 16

z.backward()
print(x.grad)     # dz/dx = 2y * 1 = 8 (对每个元素)
```

`torch.autograd` 在有向无环图（DAG）中记录所有操作，反向时自动应用链式法则。

```python
# 计算 dy/dx: y = x^2 + exp(x), x=1.0
x = torch.tensor(1.0, requires_grad=True)
y = x**2 + torch.exp(x)
dy_dx = torch.autograd.grad(y, x)[0]
print(dy_dx)  # 2*x + exp(x) at x=1 = 2 + e ≈ 4.718
```

**注意**：反向计算图在 `backward()` 完成后默认释放，需保留需指定 `retain_graph=True`

自动微分还可以记录 Python 控制流（if、while），计算图随运行时动态构建。

---

## PyTorch: Tensor 进阶

### Tensor 索引

```python
import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(t[0])        # 第0行
print(t[:, 1])     # 第1列
print(t[1:3, 0:2]) # 切片

# 多维索引遵循 Python/NumPy 规则
# 索引从0开始，负索引从末尾计，区间左闭右开
```

### 特殊张量

- **Irregular Tensor（不规则张量）**：某个轴上元素数量可变，使用 `torch.nested.nested_tensor`
- **Sparse Tensor（稀疏张量）**：稀疏数据（如宽嵌入空间），使用 `tensor.to_sparse()`

---

## PyTorch: Module / Layer / Model

### nn.Module 基类

所有神经网络模块的基类，核心方法：
- `__init__()`：定义并初始化网络层
- `forward()`：定义前向传播（计算流程）
- `parameters()`：自动追踪可学习参数

```python
import torch
import torch.nn as nn

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = SimpleModule()
x = torch.tensor([0.0288, -0.3256, 0.5925])
output = model(x)
```

### 常见网络层类型

| 层类型 | 说明 | PyTorch API |
|:-------|:-----|:------------|
| 线性层 | 全连接，特征映射 | `nn.Linear(in, out)` |
| 卷积层 | 图像处理，空间特征提取 | `nn.Conv2d(in_ch, out_ch)` |
| 激活函数 | 引入非线性 | `nn.ReLU()`, `nn.Sigmoid()` |
| 循环层 | 处理序列数据 | `nn.RNN()`, `nn.LSTM()` |
| 池化层 | 降维，减少参数 | `nn.MaxPool2d(kernel_size)` |
| 展平层 | 多维转一维 | `nn.Flatten()` |
| Dropout | 减少过拟合，随机丢弃 | `nn.Dropout(p=0.5)` |
| BatchNorm | 批归一化，稳定训练 | `nn.BatchNorm2d(channels)` |

### 批归一化（Batch Normalization）

- 放在 ReLU 激活层**之前**
- 将每个通道数据调整为固定均值和方差
- 作用：训练更稳定（可用更大学习率）、抑制梯度消失、正则化减少过拟合

### Dropout 层

- 超参数：`keep_prob` 保留率（或 `drop_rate = 1 - keep_prob`）
- 保留的权重乘以 $\frac{1}{\text{keep\_prob}}$，确保前后总和不变

### 从层到模型

```python
# 使用 nn.Sequential 快速构建
model = nn.Sequential(
    nn.Linear(3, 3),
    nn.ReLU(),
    nn.Linear(3, 2),
    nn.Softmax(dim=1)
)

# 查看模型子模块和参数
for name, child in model.named_children():
    print(name, child)
for name, param in model.named_parameters():
    print(name, param.shape)
```

---

## PyTorch: 模型训练流程

```python
import torch
import torch.nn as nn

# 1. 定义模型
model = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 2))

# 2. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. 训练循环
for epoch in range(100):
    optimizer.zero_grad()       # 清空梯度
    output = model(X_train)     # 前向传播
    loss = criterion(output, y_train)  # 计算损失
    loss.backward()             # 反向传播
    optimizer.step()            # 更新参数
```

---

## PyTorch: Transforms 数据变换

数据预处理的桥梁：将原始数据转换为模型可接受的格式。

- `transform`：处理特征（如图像归一化）
- `target_transform`：处理标签（如类别转编码）

### ToTensor

将 PIL 图像或 NumPy 数组转换为 FloatTensor，并将像素值从 $[0, 255]$ 归一化到 $[0.0, 1.0]$

### Lambda 变换

自定义任意变换函数，常用于标签独热编码：

```python
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch

ds = datasets.FashionMNIST(
    root="data", train=True, download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
    )
)
```

### Compose 组合变换

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

---

## PyTorch: 模型保存与加载

```python
# 推荐：只保存参数（state_dict）
torch.save(model.state_dict(), 'model.pth')

# 加载
model = MyModel()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()  # 评估模式

# 训练检查点（断点续训）
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

**三种保存方式对比：**

| 方式 | 优点 | 缺点 |
|:-----|:-----|:-----|
| 整个模型 | 简单直接 | 文件大，依赖类定义 |
| state_dict（推荐） | 文件小，灵活 | 加载前需创建相同架构 |
| 检查点 | 支持断点续训 | 文件最大 |

---

## 全连接网络 (FCN) 与图像分类

### FCN 结构特点

- 全连接（Dense）：相对于稀疏（Sparse）连接
- 多层感知机（MLP）是最基本的全连接网络
- 与卷积网络（局部连接）不同，FCN 每层所有神经元互相连接

### CIFAR10 数据集实战

```python
import torch
import torchvision

# 包含 10 类，共 60000 张彩色图片（50000 训练 + 10000 测试）
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
```
