# Mi0331学习记录

## Optimize parameters with Pytorch

Forward propagation, calculate loss, backpropagation, parameter update, iteration.

Hyperparamaters:
- learning_rate
- batch_size
- epochs

Loss functions:
- nn.MSELoss (均方误差)
- nn.NLLLoss（负对数似然）
- nn.CrossEntropyLoss （结合nn.LogSoftmax & nn.NLLLoss）

Optimizers:
- SGD
- Adam （自适应学习率）
- RMSprop （适应+RNN）
- Adagrad （参数独立学习率+sparse data）

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    pass
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred,y)

        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(...)
```

```python
def test_loop(dataloader, model, loss_fn):
    model.eval()
    pass

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
```

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

epochs = 10
for t in range(epochs):
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
```

## CNN Multilayer Convolutional Networks

### Structure of multi-layer network

connection pattern inter-layer

fully connected network

feedforward(MLP, CNN), feedback(RNN), memory network

### Terminology

- Convolution: Convolution operation is applying convolution kernel to input nd-array data, such as a image
- kernel/filter: Feature extraction in image with different conv. layers. The kernel/filter is a nd-array. The elements in Kernel is learned by ML. 
- no padding
- padding
- zero-padding
- s stride
- channel
- subsample/down sampling
- pooling (max-pooling, avg-pooling)
- normalization layers
- batch normalization
- dropout layer

### Kernel

```python
def convolve(input, kernel):
    h_in, w_in = input.shape
    h_k, w_k = kernel.shape

    h_out = h_in - h_k + 1
    w_out = w_in - w_k + 1

    output = np.zeros((h_out,w_out))

    for i in range(h_out):
        for j in range(w_out):
            window = input[i:i+h_k, j:j+w_k]
            output[i,j] = np.sum(window * kernel)
    
    return output

input = np.array([[2,2,2,2],[4,4,4,4],[8,8,8,8]])
kernel = np.array([[1,1],[2,2]])

output = convolve(input, kernel)
```

### Padding & Stride

### Channel

Input 3D tensor with multiple convolution kernels.

### Activation function layer

Non-linearity activation, ReLU, Sigmoid

### Subsample layer

subsample reduce the size of output tensor.

- 最大池化（max-pooling），取其中最大值 pick up the maximum value
- 平均池化（avg-pooling），取其中的平均值 pick up the average value

### Batch Normalization Layer

Batch normalization is always placed before the ReLU activation layer, it output the values with fixed average and deviation.

- Make training process more stable, even with a larger learning rate
- suppress gradient vanish problem 
- Regularization, reduce overfitting

### Dropout Layer

**Reduce the overfitting problem.**

hyperparameter： keep_prob or drop_rate=(1-keep_prob)

For all weights within the Layer, the original value is multiply by 1/keep_prob, or set to zero (dropped) otherwise.

### Conclusion

Locally connected, rather than dense & fully connected.

Advantages: 参数少 可并行操作 计算快

### Practice CNN on Pytorch


## Transformer

### Terminology

- Self attention
- Scaled dot product attention
- Positional encoding
- Masking
- Multi-head attention
- Masked language model
- BERT: Bidirectional Encoder Representations from Transformers
- Roberta: Robustly optimized BERT approach
- T5: Text-to-Text Transfer Transformer
- BART: Bidirectional and Auto-Regressive Transformers
- GPT: Generative Pre-trained Transformer

### Introduction

Transformer是一种encoder-decoder结构（Seq2Seq结构）

1. 编码器encoder: 
   - 多头自注意力层(A multi-head self-attention layer)
   - 全连接前馈网络(A fully connected feed-forward layer)
2. 解码器decoder：解码器与编码器的不同之处是，解码器有两个注意力子层
   - （掩码）多头自注意力层(Masked multi-head self-attention layer)
   - 解码器与编码器注意力层(Encoder-decoder attention layer)

**Attention is all you need**

Embeddings, Attention, Encoder-Decoder

### Self-attention (Scaled Dot-Product Attention)

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence

*Q(query), K(key), V(value)*

$Attention(Q,K,V) = softmax(sim(Q,K))V$

$sim(Q,K) = \frac{QK^T}{\sqrt{d_k}}$

### Multi-head attention

Input: Q,K,V

多头注意力由四部分组成：
- Q、K、V通过并排的多个线性（Dense）层成为多个头。
- 每个头通过按比例缩放的点积注意力模块。
- 多个头串联成一体。
- 最后一层线性层。

### Positional encoding

- 位置编码，为模型提供一些关于单词在句子中相对位置的信息。
- 位置编码向量被加到嵌入（embedding）向量中。
- 当加上位置编码后，词将基于它们含义的相似度以及它们在句子中的位置，在 d 维空间中离彼此更近。

$\begin{cases}
PE_{pos,2i}=sin(pos/10000^{\frac{2i}{d_{model}}}) \\
PE_{pos,2i+1}=cos(pos/10000^{\frac{2i}{d_{model}}})
\end{cases}$

### Masking

- Fill masking 填充值 0 出现的位置：在这些位置 mask 输出 1，否则输出 0。
- Look-ahead masking 用于遮挡一个序列中的后续标记（future tokens）。换句话说，该遮挡mask 表明了不应该使用的条目。

### Encoder layer

Each encoder layer have sub-layers: multi-head attention + point wise feed forward networks

**Residual Connection** + Layer normalization

- 残差连接有助于避免深度网络中的梯度消失问题。
- 每个子层的输出是 LayerNorm(x + Sublayer(x))。
- 归一化是在 d_model（最后一个）维度完成的。

### Decoder layer

Each encoder layer have sub-layers: multi-head attention(masking) + multi-head crossing attention + point wise feed farward networks

Residual Connection + Layer Normalization

### ViT Vision Transformer

