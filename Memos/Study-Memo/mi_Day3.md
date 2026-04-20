# Mi0310学习记录

## Assignments
1. Calculate $\pi$ in 4 ways:
   
   (1)Taylor (2)Monte-carlo (3)Chudnocsky (4)迭代

2. 

## The Fundamental of Machine Learning

$\textbf{Machine Learning}$: 

1. Supervised Learning(Regression), Unsupervised Learning(Clustering, Associated Rules), Semi-supervised Learning, Reinforcement Learning

2. Online learning(dynamic update dataset), Batch learning

3. Model-based Learning, Instance-based Learning

$\textbf{Challenges}$:

1. Data: Quantity & Quality
2. ...

Tasks: Classification & Regression

Dataset: Training, Validation, Test

To help uncover less obvious patterns by using ML (data mining)

$\textbf{Supervised Learning}$:

    在给定输入和输出对（x，y）情况下，学习映射f：x -〉y

    labeled dataset(labeled example)

$\textbf{Deep Learning}$:

    multilayers of neural network, belongs to Supervised Learning

$\textbf{Transfer Learning}$:

$\textbf{Unsupervised Learning}$:

    K-means, Gaussion Mixture Model


## Scikit-learn
```python
from sklearn import datasets, linear_model

model = linear_model.LinearRegression()
model.fit(X,y) #input X and its labels y
model.evaluate(X',y')
model.predict(X)
```

---
### Sklearn in Supervised Learning
GNB LR DecisionTree SVM
1. data preparation
from sklearn.datasets import 
2. data visualization

#### GNB 高斯朴素贝叶斯分类
   Bayes formula:

   $p(H|D)=\frac{p(H)p(D|H)}{p(D)}$
   
    Naive Bayes Classifier -> Gaussian Naive Bayes

    model.predict_proba() #查看对不同标签预测的概率

    假设不同特征之间相互独立且每个特征符合正态分布 后验概率=先验概率*由统计得概率

#### Logistic Regression:
   
   sigmoid function: $sigmoid(x) = \frac{1}{1+e^{-x}}$

   输入特征X 经过权重W 输出概率Y

#### Decision Tree:
   
   一种基于特征的分类方法

$\textbf{不确定性度量}$：

$H = -\sum_{i=1}^{n} p_i \log(p_i)$

$Gini = 1 - \sum_{i=1}^{n} {p_i}^2$

    Information Gain 信息增益：增加信息使熵减少

ID3算法：在根部选择最大的期望信息增益的特征，往后不断搜索最大的信息增益特征

#### Support Vector Machine:
    
    找出不同类之间的最宽通道，支撑向量指通道的边缘部分

    对于m个特征的分类是基于m-1维超平面

    不同的核函数（Kernel）：linear，Polynomial，RBF，sigmoid，tanh

    SVC：SVM Classification SVR：SVM Regression

    ```python
    from sklearn import svm
    model = svm.SVC(kernel='linear')
    ...
    ```

    SVM和最小二乘法的核心区别？

    最小二乘法：最小话误差平方和 SVM：最大化间隔



---
### Data Preprocessing

Feature Transform
```python
from sklearn import preprocessing
```

1. Scaling:
   
   Z-score: $\frac{Value-Mean}{SD}$

   Min-Max: $\frac{Value-Minimum}{Maximum-Minimum}

2. Continuous value transformation:

   StandardScaler (将数据转化为给定均值和标准差的正态分布)

   MinimaxScaler
   
   L1 Normalizer 权值向量中元素绝对值之和

   L2 Normalizer 各元素平方和再求平方根

3. Discrete value transformation:
   
   Binarizer 二值化

   KBinsDiscretizer k个等宽的bins将特征离散化

   np.digitize(x, bins, right=False) bins=[-1,0,1] bins中为分割点

### Dataset Splitting
```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=0) #test_size为测试集占比

model = SVC(kernel='linear')
model.fit(train_x, train_y)
pred_test = model.predict(test_x)
accuract_score(test_y, pred_test)
```

---
### Sklearn in Unsupervised Learning

#### K-means

$\arg\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} (x - \mu_i)^2$

```python
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=250, centers=4, random_state=500, cluster_std=1.25)

model = KMeans(n_clusters=4, random_state=0)
model.fit(X)
...
```

先找k个中心点 然后按照其他点距离这些点的远近分为k组 每一组再求中心点 然后重复此过程 最后一定收敛 但是不一定保证全局最优

#### GMM Gaussian Mixture Model

核心假设：数据是由k个高斯分布混合而成 分为E步和M步

E步：猜这些点分别对应k个类的概率 看离椭圆中心的远近（密度大小）

M步：根据概率分布更新椭圆参数

“椭圆来自于协方差矩阵”

```python
from sklearn.mixture import GaussionMixture

model = GaussionMixture(n_components=4, random_state=0)
model.fit(X)
...
```