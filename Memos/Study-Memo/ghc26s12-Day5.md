## PyTorch的Transforms

### 什么是Transform

- transform用于修改特征（Features），如图像的标准化、裁剪
- target_transform用于修改标签（Labels），将类别名称映射为数字编码

所有的TorchVision数据集都提供了这两个接口

### ToTensor变换

```py
from torchvision import transforms
transform = transforms.ToTensor()
img_tensor = transform(img)
```

- 将PIL Image或NumPy ndarray转换为PyTorch特有的Tensor数据格式

- 自动将像素值从[0,255]线性缩放到[0.0,1.0]的浮点数范围，同时维度顺序变化H × W × C  →  C × H × W

### Lambda变换

```python
from torchvision.transforms import Lambda
from torchvision import transforms
target_transform = transforms.Lambda(lambda y: torch.nn.functional.one_hot(torch.tensor(y), num_classes=10)) # 独热编码
transforms.Lambda(lambda x: x.flip(-1))  # 水平翻转

```

- 可以用来将整数标签转换为独热编码张量
- 允许嵌入任何变换函数



## 梯度下降法

### 定义

一种迭代搜索局部极小值的最优化算法，也称最速下降法

### 原理

实值函数在点a处可微，则沿梯度相反的方向下降最快

权重更新的公式$\theta=\theta-\eta\cdot\nabla_{\theta}J(\theta)$其中η为学习率

### 随机梯度下降法（SGD）

- 随机初始化网络的权重和偏移
- 选取随机样本，前向计算网络的各层输出和Loss
- 反向传播逐层计算偏导
- 根据学习率更新权重
- 继续下一个随机样本呢





## 反向传播（BP）

### 定义

计算损失函数对内部权重的梯度值，基于梯度下降法最小化损失函数

### 步骤

- 前向传播计算缓存节点输出值
- 反向传播遍历图，链式法则计算损失函数值对每个参数的偏导
- 局部梯度用局部缓存值计算，再和上游梯度相乘，接着向下传播

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260325091448715.png" alt="image-20260325091448715" style="zoom:33%;" />

## 神经网络的训练

### 小批量训练（mini-batch）

- 整个训练集称为一个batch，切分的子集称为mini-batch
- 介于批量训练（一次训练整个数据集）和SGD（单样本）间
- 训练完一个mini-batch称为一次iteration；训练完一整个batch称为一个epoch

| 方法       | batch size |
| ---------- | ---------- |
| SGD        | 1          |
| Mini-batch | 2 ~ N-1    |
| Batch GD   | N          |

### 训练结果

- 网络模型的层间连接结构确定
- 参数固化（frozen），每层的内部权重模型参数固定

### 应用

对训练集外的数据进行推断/预测

### 防止过拟合

- L1
- L2
- 丢弃正则化（dropout）：每次训练随机关闭部分神经元，测试时不丢弃，做缩放

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260325215803917.png" alt="image-20260325215803917" style="zoom:33%;" />

## 自动微分Autograd

### 计算图

除非显式指定保留，反向计算图在计算完毕后会立即被释放

`retain_graph`：多次反向传播同一张图

```py
# 1) retain_graph=True
# 同一计算图上做两次 backward
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

y.backward(retain_graph=True)   # 第一次反向传播，但保留计算图
print(x.grad)                   # tensor(4.)

y.backward()                    # 第二次反向传播
print(x.grad)                   # tensor(8.)，梯度累加了

```

`create_graph`：对梯度继续求导

```py
# 2) create_graph=True
# 先求一阶导，再对一阶导继续求导（二阶导）
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]  # 一阶导
grad2 = torch.autograd.grad(grad1, x)[0]                 # 二阶导
print(grad1)   # tensor(12.)
print(grad2)   # tensor(12.)

```

### 微分方法

#### 数值微分

- 舍入误差

- 效率低

#### 符号微分

直接得到求导结果表达式

#### 自动微分

| 维度       | 前向模式                                         | 反向模式             |
| ---------- | ------------------------------------------------ | -------------------- |
| 传播方向   | 输入 → 输出                                      | 输出 → 输入          |
| 计算目标   | 单个输入的导数（一次前向获得Jacobian矩阵的一列） | 所有输入的梯度       |
| 时间复杂度 | O(输入维度)                                      | O(输出维度)          |
| 内存开销   | 低                                               | 高（要存计算图）     |
| 适合场景   | 输入少                                           | 输出少（如标量损失） |

下面是基于你给的模板整理的一个更完整、结构化的 PyTorch 模块总结（Markdown 格式）：

------

## PyTorch模块

### 容器类（Containers）

用于组合和管理多个子模块，构建复杂网络结构。

- `nn.Sequential`
  - 按顺序堆叠模块
  - 适用于简单前向传播结构
- `nn.ModuleList`
  - 类似 Python list，用于存储模块
  - 不自动定义前向传播
- `nn.ModuleDict`
  - 类似 dict，按 key 存储模块
- `nn.ParameterList`
  - 存储参数列表
- `nn.ParameterDict`
  - 存储参数字典

------

### 常见层（Layers）

#### 线性层（全连接）

- `nn.Linear(in_features, out_features)`
- 用于特征映射：
  ( y = xW^T + b )

#### 激活函数层

- `nn.ReLU`
- `nn.Sigmoid`
- `nn.Tanh`
- `nn.LeakyReLU`
- `nn.Softmax`

#### 卷积层

- `nn.Conv1d`
- `nn.Conv2d`
- `nn.Conv3d`

#### 循环层

- `nn.RNN`
- `nn.LSTM`
- `nn.GRU`

#### 池化层

- `nn.MaxPool2d`
- `nn.AvgPool2d`
- `nn.AdaptiveAvgPool2d`

#### 展平层（多维 → 一维）

- `nn.Flatten`

#### 归一化层

- `nn.BatchNorm1d / 2d / 3d`
- `nn.LayerNorm`
- `nn.GroupNorm`

#### Dropout层

- `nn.Dropout`
- 防止过拟合

------

### 模块与层的关系

- 模块（Module）是 PyTorch 中的基本构建单元
- 层（Layer）本质上是**特殊的模块**
- 一个模块可以包含多个子模块（层）
- 支持嵌套组合，形成复杂模型结构

------

### 模块类（核心：nn.Module）

所有神经网络模型的基类。

#### 基本结构

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
```

#### 核心方法

- `__init__()`：定义层结构
- `forward()`：定义前向传播
- `parameters()`：获取模型参数
- `to(device)`：移动到 GPU/CPU
- `train()` / `eval()`：切换模式

------

### 学习参数类（Parameters）

用于表示需要训练的参数。

#### nn.Parameter

- 本质：Tensor 的子类
- 会自动加入模型参数中

```python
self.weight = nn.Parameter(torch.randn(10, 10))
```

#### 特点

- 自动参与反向传播
- 被 `model.parameters()` 收集
- 默认 `requires_grad=True`

------

### 模型类（Model）

模型是由多个模块组合而成的完整网络。

#### 构建方式

1. 继承 `nn.Module`
2. 定义层（模块）
3. 实现 forward

#### 示例

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)
```



## 全连接网络的PyTorch构建

### 多层神经网络

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260325084337957.png" alt="image-20260325084337957" style="zoom:33%;" />

全连接称为密集连接（dense），否则就是稀疏（如卷积网络）

### 网络拓扑结构

1. 前馈：MLP、CNN
2. 反馈：RNN、LSTM
3. 记忆：Transformer

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260325213843755.png" alt="image-20260325213843755" style="zoom:33%;" />

### 特殊层

- softmax层：转化为概率向量
- 批归一化层（对每个通道减去均值除以标准差）
- dropout layer：减少过拟合（设置drop-rate）

### 多层感知机MLP

由多层全连接组成

#### 例：FCN

```python
import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 Sequential 按顺序堆叠各层
        self.net = nn.Sequential(
            # 第1个全连接层：
            # 输入是 28*28=784 维（将图片拉平成向量后）
            # 输出是 512 维特征
            nn.Linear(28 * 28, 512),
            # ReLU 激活函数，引入非线性
            nn.ReLU(),
            # Dropout：训练时随机丢弃 20% 神经元，防止过拟合
            nn.Dropout(0.2),
            # 第2个全连接层：
            # 将 512 维特征映射到 256 维
            nn.Linear(512, 256),
            # 再次使用 ReLU 激活
            nn.ReLU(),
            # 再次 Dropout，丢弃比例仍为 20%
            nn.Dropout(0.2),
            # 输出层：
            # 将 256 维映射到 10 维，通常表示 10 个类别
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor):
        # 将输入展平：
        # 如果输入 x 的形状是 [batch_size, 1, 28, 28]
        # 或 [batch_size, 28, 28]
        # 展平后变成 [batch_size, 784]
        x = x.flatten(start_dim=1)
        # 将展平后的向量送入全连接网络
        return self.net(x)

```

#### 例：CNN

```py
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ======================
        # 特征提取部分（卷积层）
        # ======================
        self.features = nn.Sequential(
            # 第1层卷积：
            # 输入通道=1（灰度图），输出通道=32
            # 卷积核3x3，padding=1 → 保持尺寸不变（28x28）
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),

            # 最大池化：2x2 → 尺寸减半（28→14）
            nn.MaxPool2d(2),

            # 第2层卷积：
            # 输入32通道 → 输出64通道
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            # 最大池化：14→7
            nn.MaxPool2d(2),

            # 第3层卷积：
            # 输入64 → 输出128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        # ======================
        # 分类器（1x1卷积）
        # ======================
        # 相当于对每个空间位置做一个全连接（通道映射）
        self.classifier = nn.Conv2d(128, 10, 1)

    def forward(self, x: torch.Tensor):
        # 输入 x 形状：
        # [batch_size, 1, 28, 28]
        # 提取特征
        x = self.features(x)
        # 此时形状：
        # [batch_size, 128, 7, 7]
        # 分类（通道数变为10）
        x = self.classifier(x)
        # 形状：
        # [batch_size, 10, 7, 7]
        # 全局平均池化（Global Average Pooling）
        # 对每个通道取空间平均 → 去掉空间维度
        x = torch.mean(x, dim=(2, 3))
        # 形状：
        # [batch_size, 10]
        # 输出 logits（未经过 softmax）
        return x

```

