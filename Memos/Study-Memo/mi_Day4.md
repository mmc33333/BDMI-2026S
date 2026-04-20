# Mi0317学习笔记

## Deep Learning

### Pytorch Playground

playground.tensorflow.org
（这个网站不错可以有training可视化！）
---
### Artificial Neurons

#### What is it?

A linear weighted superposition of a set of inputs.

$\textbf{Activation Function:}$

1. sigmoid $\sigma (x)=\frac{1}{1+e^{-x}}$
2. tanh
3. ReLU $ReLU(x)=max(x,0)$

#### Ability of a single neuron

Boolean operation: AND OR NOT NAND XOR

characteristics of the sigmoid function: $\sigma(5)=1,\sigma(-5)=0$\\

通过控制参数b的大小来实现不同逻辑运算（0 1）

#### To solve XOR problem

不是线性分类问题，无法用单个神经元解决！

用双层神经网络来实现

---
### Multi-Layer Neuron Network

AutoDiff, Back propagation, Weight updates, Gradient Descent, Stochastic Gradient Descent

#### Structure

```python
import numpy as np
import torch

```

#### Softmax & Logit Function

Softmax processing of the output layer calculates a probability distribution vector.

Logit: 把区间(0,1)内的数值，变换到区间(-∞,﹢∞)
Convert the value in the interval (0,1) to the interval (-∞,+∞)
```python
import numpy as np

def logit(int x){
    if x<0 or x>1:
        return -1
    return np.log(x/(1-x))
}
```

#### Update weights

1. Define the loss function
2. Initialization: Random Initialization
3. Back Propagation: Calculate the gradient of the loss function to the weights
4. Weights Adjusting: Stochastic gradient descent

#### Numerical function of loss-metric function

1. Absolute Error
2. Mean Absolute Error (MAE)
3. Squared Error
4. Mean Squared Error (MSE)
5. Cross Entropy

The goal of training is to minimize the difference between the actual output of the network and the expected output (i.e., labels) after the training data (samples) are fed into the network by adjusting the internal weights of the network

L1 Loss Function: actual value

L2 Loss Function(squared loss): Because of the squared value, this loss function amplifies the impact of poor predictions. That is, the squared loss function reacts more strongly to outliers than the L1 loss function.

Regression: MSE, Classification: CE

```python
#Calculate MSE
import numpy as np
x = np.array([72,94,79,83,65,81,73,67,85,82])
x_mean = np.sum(x)/len(x)
x_MSE = np.sqrt(np.mean((x-x_mean)**2))
```

```python
#Calculate CE
import numpy as np
#assume y_pred & y_true is ready
H = -np.sum(y_true * np.log(y_pred))

#to avoid nan
epsilon = 1e-15
y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
H_safe = -np.sum(y_true * np.log(y_pred_clipped))
```

### Backpropagation Algorithm

#### Gradient Descent Method

(没啥好说的 梯度下降)

#### Back Propagation BP

1. 输入数值进行前向计算获得loss， loss=D(y, y’)
2. 根据损失函数的性质以及链式求导法则
3. 反向逐层计算损失函数对权重的梯度（各个偏导数）

layer-by-layer calculation of the gradient of the loss function with respect to the weights in reverse direction (individual partial derivatives)

#### Gradient Descent Optimization Algorithms

• Momentum (important)
• Nesterov accelerated gradient
• Adagrad
• Adadelta
• RMSprop
• Adam (important)
• AdaMax
• Nadam
• AMSGrad

#### Stochastic Gradient Descent Method

• step 1 randomly initialize the weights
• step 2 choose a random sample
• Step 3 Generate a output and from the loss calculates its gradient to each
weights backward
• Step 4 Adjusting the weights in each layer
• Go to step 2, continue to choose a random sample next.

不用遍历所有数据再更新，而是随机样本然后更新，提高了更新速率但粗糙

折中方案：mini-batch

#### Mini-batch Gradient Descent

### Overfitting Problem & Regulatization

L1, L2, Drop regularization



---
## Pytorch Basic Knowledge

tensor, variable, automatic differentiate, module, computational graph, training process

### Tensor

Multi-deimensional arrays with a uniform.

All tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.

Tensor shape: shape rank axis dimension size

1. Tensor Creation:
```python
import torch

rank0_tensor = torch.tensor(4)
rank1_tensor = torch.tensor([2.0,3.0,4.0])
rank2_tensor = torch.tensor([[1,2],[3,4],[5,6]],dtype=torch.float16)
```

2. Tensor Operation:
```python
torch.add()
torch.multiply() #(corresponding elements)
torch.matmul()

import torch.nn.functional as F

torch.max()
torch.argmax() #find the index of the max element
F.softmax(x, dim=1) #normalize the tensor along specified axis
```

3. Dtypes
```python
f16_tensor = f64_tensor.to(torch.float16)
u8_tensor = f64_tensor.to(torch.uint8)
```

4. Tensor Reshape:
```python
#torch.reshape(tensor, shape)
a = torch.tensor([[1],[2],[3]])
b = torch.reshape(a,[1,3])

#Typically the only reasonable use of tf.reshape is to combine or split adjacent axes.
```

5. Tensor Broadcasting
Smaller tensors are stretched automatically to fit larger tensors when running conbined operations on them
```python
x = torch.tensor([1,2,3])
y = torch.tensor(2)

ans = torch.multiply(x,y) #broadcast here
```
The simpliest and most common case is when attempting to multiply or add a tensor to a scalar.
```python
x = torch.tensor([1,2,3])
x = x.reshape([3,1])
y = torch.arrange(1,5)
print(x*y)
# tensor([[1,2,3,4],
#    [2,4,6,8]
#    [3,6,9,12]],device='cuda:0')
```

## Logistic Regression: Binary Classification

### Performance Indicators:

1. Accuracy = $\frac{TP+TN}{TP+TN+FP+FN}$ 预测中对的占比
2. Precision = $\frac{TP}{TP+FP}$ 预测中的有多少对了
3. Recall = $\frac{TP}{TP+FN}$ 预测的在真实此类中有多少对了
4. F1-score = $\frac{2*Precision*Recall}{Precision+Recall}$

### Logistic Regression Practice - From manual to automated

1. Get training data
2. Build the model
3. Define the loss function
4. Run the training data and calculate the loss the target value
5. Calculate the gradient of the loss from the target value
6. Calculate the gradient of the loss and use an optimizer to adjust the variables to fit the data
7. Outcome evaluation

```python
import pandas as pd
import numpy as np

data = pd.read_excel('data.xlsx')
data.head()

data = data.rename(columns={'c1': 'label'
                            'c2': 'height'
                            'c3': 'weight'
                            })
data['label'] = data['label'].apply(lambda x:{'男':0, '女':1})

#normalized
features = data[['height','weight','hair']].to_numpy()
mean = np.mean(features,axis=0)
std = np.std(features,axis=0)
features = (features - mean)/std

label = data['label'].to_numpy

#build the model
def sigmoid(scores):
    return 1/(1+np.exp(-scores))

#中间这段似然 重新整理

def negative_log_likelihood(features, target, weights):
    scores = np.dot(features,weights)
    l1 = np.sum(target*scores - np.log(1+np.exp(scores)))
    goal = -l1 #maximum likelihood
    return goal

def logistic_regression(features, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0],1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        output_error_signal = target - prediction

        gradient = np.dot(features.T, output_error_signal)
        weights = weights + learning_rate * gradient

    return weights

weights = logistic_regression(features, label, num_steps = 50000, learning_rate = 5e-5, add_intercept=True)

def predict(features, weights):
    global mean
    global std
    features = (features - mean)/std
    intercept = np.ones((features.shape[0],1))
    features = np.hstack((intercept, features))
    scores = np.dot(featrues, weights)
    predictions - sigmoid(scores)

    return predictions
```