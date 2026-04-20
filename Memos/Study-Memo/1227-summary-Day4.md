## 20260317 BDMI课程小记
#### by 物理42 俞善斌
第四次课。
- 利用TensorFlow Playground对**TensorFlow**以及相关的机器学习概念进行了介绍。  
- 介绍了**二元分类问题**以及**逻辑斯提回归**的二元分类方法，包括问题的各项参数
- 介绍了**人工神经元**（单层神经网络）的基本结构及其模拟布尔运算（AND、OR、NOT、NAND）的能力，介绍了其在XOR问题上的局限性。
- 介绍了**多层神经网络**的原理，主要包含以下方面：  
  - 张量表示与网络结构  
  - 激活函数，损失函数
  - 反向传播算法与梯度下降权重更新  
  - 小批量训练、过拟合与正则化
- 介绍了**PyTorch**基础知识，包括张量的创建、形状、数据类型、运算、重塑、广播以及张量与NumPy的转换。

---

### 二元分类问题与逻辑斯提回归

#### 二元分类问题

- **概念**：二元分类是监督学习中的一类经典任务，目标是将样本划分为两个互斥的类别（通常标记为0和1，或正类和负类）。模型学习从输入特征到输出类别的映射关系，输出可以是离散的类别标签，也可以是样本属于正类的概率值。

- **评价指标**：二元分类模型的性能通过混淆矩阵中的四个基本量进行评估：
  - **TP（真正例）**：实际为正类，预测为正类的样本数
  - **TN（真负例）**：实际为负类，预测为负类的样本数
  - **FP（假正例）**：实际为负类，预测为正类的样本数（第一类错误）
  - **FN（假负例）**：实际为正类，预测为负类的样本数（第二类错误）

  基于上述四个量，定义以下评价指标：
  - **准确率（Accuracy）**：预测正确的样本占总样本的比例  
    \[
    \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
    \]
  - **精确率（Precision）**：预测为正类的样本中实际为正类的比例  
    \[
    \text{Precision} = \frac{TP}{TP + FP}
    \]
  - **召回率（Recall）**：实际为正类的样本中被正确预测的比例  
    \[
    \text{Recall} = \frac{TP}{TP + FN}
    \]
  - **F1-score**：精确率与召回率的调和平均数，综合评估模型性能  
    \[
    \text{F1-score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
    \]

#### 逻辑斯提回归（Logistic Regression）

- **模型结构**：逻辑斯提回归采用单个人工神经元结构，将输入特征的线性加权组合通过Sigmoid激活函数映射到(0,1)区间，输出样本属于正类的概率。

- **数学表达式**：
  - **线性组合**：  
    \[
    z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n = \beta^T \mathbf{x}
    \]
    其中 \(\beta_0\) 为偏置（截距），\(\beta_1, \dots, \beta_n\) 为特征权重，\(x_1, \dots, x_n\) 为输入特征。

  - **Sigmoid激活函数**：将线性组合映射为概率值  
    \[
    p(x) = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(\beta^T \mathbf{x})}}
    \]
    其中 \(\sigma(z) \in (0,1)\)，输出可解释为样本属于正类的概率。

  - **分类决策**：通常以0.5为阈值  
    \[
    \hat{y} = \begin{cases} 
    1, & p(x) \geq 0.5 \\ 
    0, & p(x) < 0.5 
    \end{cases}
    \]

- **损失函数**：采用负对数似然损失（交叉熵损失），目标是最小化预测概率与真实标签之间的差异  
  \[
  \text{Loss} = -\sum_{i=1}^{n} \left[ y_i \log p(x_i) + (1 - y_i) \log (1 - p(x_i)) \right]
  \]

- **权重更新**：通过梯度下降法最小化损失函数，梯度计算为  
  \[
  \nabla_{\beta} \text{Loss} = \sum_{i=1}^{n} (y_i - p(x_i)) \mathbf{x}_i
  \]
  权重更新公式为  
  \[
  \beta_{\text{new}} = \beta_{\text{old}} + \eta \cdot \nabla_{\beta} \text{Loss}
  \]
  其中 \(\eta\) 为学习率（步长）。

---

### 多层神经网络

#### 网络结构与张量表示
- 多层神经网络由输入层、若干隐藏层和输出层构成，每层包含若干人工神经元。
- 网络的权重和输入输出均可表示为**张量**（Tensor）：
  - 输入 \(\mathbf{X}\)：形状为 \([batch\_size, input\_dim]\)
  - 第 \(l\) 层权重 \(\mathbf{W}^{(l)}\)：形状为 \([input\_dim^{(l)}, output\_dim^{(l)}]\)
  - 第 \(l\) 层输出 \(\mathbf{H}^{(l)} = f(\mathbf{X}^{(l-1)}\mathbf{W}^{(l)} + \mathbf{b}^{(l)})\)

#### 前向传播与激活函数
- **前向传播**：数据从输入层逐层向前计算，直至输出层。
- **激活函数**：为网络引入非线性，常用函数包括：
  - Sigmoid：\(\sigma(x) = 1/(1+e^{-x})\)
  - tanh：\(\tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})\)
  - ReLU：\(\text{ReLU}(x) = \max(0, x)\)

#### 损失函数
- 根据任务类型选择损失函数：
  - **回归任务**：均方误差（MSE）\(\displaystyle \text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2\)
  - **分类任务**：交叉熵（Cross Entropy）\(\displaystyle H(y,\hat{y}) = -\sum y_i \log \hat{y}_i\)

#### 反向传播与权重更新
- **反向传播**：利用链式法则从输出层向输入层逐层计算损失函数对各层权重的梯度。
- **梯度下降**：按梯度的反方向更新权重，使损失函数逐步减小  
  \[
  \mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - \eta \cdot \nabla_{\mathbf{W}} \text{Loss}
  \]
  其中 \(\eta\) 为**学习率**，控制更新步长。

#### 训练策略
- **小批量训练（Mini‑batch）**：每次使用一小批样本计算梯度并更新权重，兼顾计算效率与收敛稳定性。
- **迭代次数与轮数**：
  - 一次前向+反向称为一次**迭代（iteration）**
  - 完整遍历一次全部训练数据称为一个**轮次（epoch）**

#### 过拟合与正则化
- 模型在训练集上表现优异，但在测试集上泛化能力差，称为**过拟合**。
- 常用正则化手段：
  - **L1/L2 正则化**：在损失函数中添加权重的范数惩罚项，限制权重过大。
  - **丢弃（Dropout）**：训练时随机“丢弃”一部分神经元，迫使网络学习更鲁棒的特征。

---

### PyTorch 基础知识

#### 张量（Tensor）
- **定义**：多维数组，具有统一的数据类型（dtype），是不可变对象。
- **秩（Rank）**：张量的维度数量，标量（rank‑0）、向量（rank‑1）、矩阵（rank‑2）、高维张量（rank‑3+）。
- **形状（Shape）**：各维度的长度；**大小（Size）**：总元素数，即形状各维度长度的乘积。

#### 张量创建
- 从 Python 列表或数组创建：`torch.tensor([[1,2],[3,4]], dtype=torch.float16)`
- 自动推断数据类型（整数→`torch.int32`，浮点数→`torch.float32`）

#### 数据类型与转换
- 查看数据类型：`tensor.dtype`
- 类型转换：`tensor.to(torch.float16)`

#### 张量运算
- 基本运算：`torch.add()`、`torch.mul()`（逐元素）、`torch.matmul()`（矩阵乘法）
- 支持运算符重载：`a + b`、`a * b`、`a @ b`
- 与 NumPy 转换：需先将张量移至 CPU（`tensor.cpu().numpy()`），或使用 `np.array(tensor.cpu())`

#### 重塑（Reshape）
- `torch.reshape(tensor, new_shape)`：改变张量形状，合理使用可合并或拆分相邻维度。

#### 广播（Broadcasting）
- 当两个张量进行运算时，较小的张量会自动“拉伸”至较大张量的形状（如标量与向量相乘）。

#### 设备管理
- 查看默认设备：`torch.get_default_device()`
- 设置默认设备：`torch.set_default_device("cuda")`
- 张量创建后默认位于指定设备，可通过 `.cpu()` 移至 CPU。