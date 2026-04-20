## TensorFlow PlayGround

### 分类问题的组成部分

- Features选择二维数据集划分的特征，有直线、三角函数等
- 隐藏层：可选择层数和每层的神经元数
- 输出层：观察分类效果
- epoch：迭代次数（梯度反传次数）
- learning rate：越大越容易波动，越小收敛速度越慢
- 还可选择激活函数、正则化



## 人工神经元（Artificial Neuron）

### 定义

- 一组输入的线性加权叠加再经过一个非线性变换（激活函数）
- $y = f(w^Tx+b) = f(\sum_{i=1}^NW_iX_i+b)$

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260317142210495.png" alt="image-20260317142210495" style="zoom:33%;" align='left'/>

### 激活函数

- ReLU：$max(x,0)$
- Sigmoid：$\dfrac{1}{1+e^{-x}}$（类比逻辑斯蒂回归）
- tanh：$\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$

### 逻辑运算

- XOR异或运算非线性可分，单个人工神经元无法解决
- <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260317143127520.png" alt="image-20260317143127520" style="zoom:33%;" />

## 模型性能指标

### 1. 精确率（precision）

- 预测阳性中真阳性的比例（表征误报率）

- $P = \dfrac{TP}{TP+FP}$

### 2. 召回率（recall）

- 所有阳性中预测阳性的比例（表征漏报率）

- $R = \dfrac{TP}{TP+FN}$

### 3. 准确率（accuracy）

- 准确率表示模型预测正确的样本数占总样本数的比例
- $acc = \dfrac{TP+TN}{All}$

### 4. F-1 score

- $F1-score  = \dfrac{2PR}{P+R}$

---

## 深度学习

### 全连接网络（FCNN）的分层

- 输入层
- 隐藏层
- 输出层

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260317145952223.png" alt="image-20260317145952223" style="zoom:33%;" align='left'/>

全连接和后续的卷积神经网络（非全连接）相对应

### 张量（Tensor）

- 循环嵌套的ndarray

- RGB图像是一个三维张量

- ```python
  import numpy as np
  ten = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
  ```


<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260317150134258.png" alt="image-20260317150134258" style="zoom:33%;" align='left'/>

### softmax函数

- 将多分类归一化为概率向量

- $g(z_m)=\dfrac{e^{z_m}}{\sum_ke^{z_k}}$

### 分对数logit

- sigmoid函数的反函数
- $\sigma^{-1}(x) = log (\dfrac{x}{1-x})$
- 把区间（0，1）内的数值，变换到区间（-∞，+∞）

### 权重自动更新

- 训练数据正规化（z-score）$(feature-mean)/std$
- 定义损失函数
- 权重初始化
- 反向传播：计算损失函数对权重的梯度
- 权重更新：常用随机梯度下降法
- 迭代最小化损失函数，使实际输出与预期输出之间差异最小

### 损失函数/成本函数

- L1
- L2（SSR）
- MAE：L1取平均
- **MSE**（回归用）：L2取平均
- **cross entropy/CE**（分类用）
  - 负对数似然函数
  - $H_{y'}(y)=-\sum_iy_i'log(y_i)$
  - y‘是训练样本对应的标签，y表示神经网络的输出（预测的概率向量）
  - 似然是针对过去发生的事件
  - 常见于Logistic Regression

### 梯度下降法

- 为了找到函数的局部最小值，沿可微函数当前点对应梯度的反方向，按learning rate迭代搜索
- 梯度为损失函数对内部权重的梯度
- 通过链式法则向后传播





---

## PyTorch基础

### 安装

- 先管理员身份运行命令提示符，输入nvidia-smi检查CUDA --version
- 在VScode创建的虚拟环境中pip install

### 张量

1. rank-0张量又称标量（scalar），没有轴（axes）
2. rank-1张量又称向量（vector），有一个轴
3. rank-2张量又称矩阵（matrix），有两个轴
4. 以此类推
5. 张量不可改变（immutable），只能创建新的张量

#### 术语（terminology）

- shape：每个维度的长度（元素数量）
- rank：维数
- axis：张量的一个特殊维度
- size：张量的总项数

#### tensor创建

```python
import torch
rank2_tensor = torch.tensor([[1,2],[3,4],[5,6]],dtype=torch.float16)
```

#### tensor运算

```py
torch.add(a,b) # 逐元素相加
torch.mul(a,b) # 逐元素相乘
torch.matmul(a,b) # 矩阵相乘
```

#### tensor索引

```python
import torch.nn.functional as F
torch.max()
torch.argmax() # 最大值的索引
F.softmax() # 沿某个轴做归一化
```

#### 特殊tensor表示

```py
f16_tensor = f64_tensor.to(torch.float16)
u8_tensor = f16_tensor.to(torch.uint8)
```

#### reshape

```py
rank3_tensor = torch.tensor(
[[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
[[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
[[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],]
)
a = torch.reshape(rank3_tensor, [3* 2, 5])
print(a, a.shape)

a = torch.reshape(rank3_tensor, [3, -1]) # 使用 -1 自动计算
print(a, a.shape)

# tensor([[ 0, 1, 2, 3, 4],
#	[10, 11, 12, 13, 14],
#	[15, 16, 17, 18, 19],
#	[20, 21, 22, 23, 24],
#	[25, 26, 27, 28, 29]],device='cuda:0') torch.Size([6, 5])
# tensor([[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#	[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#	[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]], device='cuda:0') torch.Size([3, 10])
```



#### broadcasting

```py
x = torch.tensor( [1, 2, 3])
y = torch.tensor (2)
z = torch.tensor( [2, 2, 2])
print(torch.multiply(x, y))
print(x * y)
print(x * z)
# tensor([2, 4, 6], device='cuda:0')
# tensor( [2, 4, 6], device='cuda:0')
# tensor ( [2, 4, 6], device='cuda:0')
```

