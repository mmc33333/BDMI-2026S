## 机器学习基本概念

### 和传统学习的区别

- 自动学习规则

### ETP定义

- 利用经验集E，计算机任务T在性能度量P上有提升

### 优势

- 分析大量数据，发现隐藏规律，也成为数据挖掘（data mining）

### 类别

- #### 是否有监督：
  
  - **有监督**：
    - 给定输入输出对（x，y），学习映射f
    - 有标记的分类任务（assign labels to features）；回归
    - 贝叶斯分类器、逻辑回归、决策树、随机森林、支持向量机
    - 深度学习/神经网络
  - **无监督**：
    - 聚类；关联规则学习；异常检测
    - PCA，自动编码器
  - **半监督**：
    - 先聚类，再根据标签分类
  - **强化**：
    - 智能体观察环境，根据策略选择动作，获得反馈，更新策略，迭代
    - 策略梯度
- #### 根据数据加载方式：
  
  - **在线学习**：数据流式输入
  - 批学习：一次性加载全部数据集
- #### 是否基于整体数据
  
  - 基于模型：
    - 从实例构建模型
    - 训练慢，预测快
  - 基于实例：
    - 用已学习的例子，根据相似度度量新的实例
    - 训练快，计算慢（需要计算新样本与其他样本的距离）

### 数据的挑战

- **数据准备问题**：
  - 数据量不足
  - 数据代表性不强（过拟合）
- **数据清洗问题**：
  - 低质量数据（错误、异常值、噪声）
  - 无关特征的数据

### 数据集的分类

- 训练集：根据损失函数迭代调整参数
- 验证集：调整超参数，k-fold交叉验证中融合在训练中
- 测试集：不可泄漏在训练中，用于评估模型性能

### 模型的训练和推断

- 训练：获得映射f
- 推断：给定输入x输出y

### 深度学习

多层神经网络（>1层隐藏层）

### 迁移学习

- 将信息从一个机器学习任务迁移到另一个机器学习任务
- 数据多——数据少；简单任务——复杂任务

## scikit-learn

### 功能

- 预处理
  - 特征变换
  - 数据集划分
- 预测
- 分类

### 数据格式

- Numpy的ndarray
- Scipy的sparse稀疏数组

### API接口

- 估计器estimator：model.fit(X,y)
- 预测器predictor：model.predict(X_test)

- 转换器transformer：数据预处理、特征提取与特征选择

### 使用流程

```python
from sklearn import datasets, linear_model
model = linear_model.LinearRegression()
model.fit(X,y)
model.evaluate(X',y')
model.predict_proba(X) # 查看对不同标签预测的概率
model.predict(X_test)
```

### 有监督学习的应用

#### 数据准备

```py
from sklearn.datasets import make_classification # 创建一组带标签的数据
```

#### 高斯朴素贝叶斯（GNB）

```py
from sklearn.naive_bayes import GaussianNB
```

- Bayes formula：$ P(H|D) = \dfrac{P(D|H) \cdot P(H)}{P(D)} $
- 利用后验信息更新先验概率以获得后验概率
- **朴素贝叶斯分类器**：
  - 假设各特征之间相互独立
  - $预测结果 \hat{y} = \arg\max_{Y} \left( P(Y) \cdot \prod_{i=1}^{n} P(x_i|Y) \right) $
- **高斯朴素贝叶斯**（GNB）：
  - 假设数据符合高斯分布

#### 逻辑斯蒂回归（Logistic Regression）

```py
from sklearn.linear_model import LogisticRegression
```

- 解决的是分类问题
- $F(x) = sigmoid(x) = \dfrac{1}{1+e^{-x}}$
- $y = F(\sum_{i=1}^Nw_ix_i +b)$

#### 决策树（Decision Tree）

```py
from sklearn.tree import DecisionTreeClassifier
```

- 基于特征的分类方法
- 不确定性的度量：
  - 熵测度：$-\sum_{i=1}^n p_ilog(p_i)$
  - Gini测度：$Gini = 1-\sum_{i=1}^np_i^2$
- 信息增益：降低不确定性
- ID3算法：
  - 在决策树的各个节点上，选择最大信息增益的特征

#### 支持向量机SVM

```py
from sklearn import svm
clf = svm.SVR()
clf.fit(X,y)
```

- 找出不同类之间最大的间隔（基于超平面）
- 可选择不同的核函数kernel
  - linear
  - polynomial
  - RBF：Radial Basis Funciton
  - sigmoid，tanh
- **特点**：
  - 适用于中小型复杂数据集
  - 输入数据需要归一化

## 数据预处理

```python
from sklearn import preprocessing
```

### 特征变换

- 将自然语言的特征变换为特征向量（比如性别变换为0，1）

- #### 缩放：

  - $Z-score = \dfrac{Value-Mean} {SD}$
  - $Min-Max = \dfrac{Value-Min} {Max-Min}$

- #### 连续值变换

  - preprocessing.StandardScalar：
    - 将数据转化为训练集中的均值、标准差决定的正态分布
  - preprocessing.MinmaxScalar：
    - 将数据缩放到给定的最大最小值之间
  - L1归一化：
    - $\dfrac{1}{\sum_{i=1}^n|x_i|}$（L1范数）
    - 利于形成稀疏矩阵
  - L2归一化：
    - $\dfrac{1}{\sqrt{\sum_{i=1}^nx_i^2}}$（L2范数）
    - 防止过拟合

- #### 离散值变换

  - Binarizer二值化为0，1
  - KBinsDiscretizer使用k个等宽bins把特征离散化

### 数据集的拆分

- ```py
  from sklearn.model_selection import train_test_split
  # 先预处理再划分
  train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=0) # 测试集占总数据的 33%
  ```

### 聚类方法

#### k-means聚类

- ```py
  from sklearn.cluster import KMeans
  ```

- n个观测点划分到k个集合，使每个集合组内平方和（WCSS）最小

- $$ \arg \min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} \| x - \mu_i \|^2 $$ ，其中 $\mu_i$ 是 $S_i$ 中所有点的均值

- 迭代计算质心位置

#### 高斯混合模型（GMM，Gaussian Mixture Model）

- ```py
  from sklearn.mixture import GaussianMixture
  ```

- 基本假设：每个簇的数据都符合高斯分布

- 每个点属于每个簇都有一定的概率，更灵活



