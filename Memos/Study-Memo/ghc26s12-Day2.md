# Python进阶

### Jupyter Notebook

- 编译环境为python3.7.2，支持numpy、matplotlib、pandas等库
- 每个cell都是独立的代码块，支持txt、md、py等文件格式，可以直接运行（Shift+Enter）并输出结果

### Conda

- conda create -n env_name python=3.10
- conda activate env_name
- conda deactivate

### Pip

- pip list：输出已下载的包
- pip install jupyter notebook
- jupyter notebook打开浏览器端的jupyter notebook

### Vscode Terminal

- win+R
- python xx.py运行python代码

## Numpy库

### 基础操作

```python
import numpy as np

# 创建向量/矩阵
c = np.array([1,2,3],[4,5,6],dtype=complex) # or np.int32\bool
a = np.arange(4)
np.identity # 创建单位矩阵

# 向量/矩阵运算
a+4
a*2 # 标量的运算都是逐元素的
A = np.arange(0,9).reshape(3,3)
B = np.ones((3,3))
A*B # 为逐元素相乘
np.dot(A,B) # 为向量内积
np.matmul(A,B) # 为矩阵乘法
Aa = A.ravel() # 转换回一维向量
A.transpose() # 转置

# 调试
array.shape
array.dtype
type(variable)
import pdb;pdb.set_trace()

```

### 随机数

```python
import numpy as np
import numpy.random as npr
npr.rand(5,5) # [0,1)均匀分布的随机数列
npr.randn(d0,d1,...,dn)标正分布，指定维度d
npr.randint(low,high, size=None, dtype)默认int
npr.standard_normal(sample_size)
npr.chisquare(df=0.5,size=sample_size)
npr.poisson(lam=1.0,size=sample_size)
```

### Monte Carlo Simulation（MCS）

随机投点计算圆周率

## Matplotlib

```python
import matplotlib.pyplot as plt
```

### kwargs

- 设置图表绘制的属性
- 关键字作为参数传递给函数
- 如linewidth、fontsize、linestyle

### scatter plot

- 离散的数据点
- text用于给点添加注释
- title直接添加图表标题
- grid添加网格
- legend添加图注，可更改loc kwarg改变位置，默认的1为右上角

### line plot

- 一条线连接数据点序列

- 可以直接将DataFrame作为参数传递给plot获得多线性图

- ```python
  plt.plot(t,y1,'b*',t,y2,'g^',t,y3,'ys')
  ```

- ```python
  plt.plot(t,y1,'b -- ',t,y2,'g',t,y3,'r -. ')
  ```


### Histograms

- hist（data，bins=20）

### Bar Charts

- x轴表示index
- xticks传递标签位置
- yerr可以传递标准差
- barh变水平
- stacked将不同样本堆叠起来

### Pie Chart

- pie（）计算每个值占用的百分比
- explode突出某一块扇形

### Others

- box plot：箱线图
- contour plot：等值线，colorbar调整色彩映射
- mplot3d：meshgrid后plot_surface

#### subplot

- 可绘制多个子图（row，col，current_im_index）
- 第一个数字确定将图形垂直分割成多少个部分
  第二个选项确定将图窗水平分割成多少个部分
  第三个数字选择我们可以在其上指示命令的当前子图

### Pygal

- 先安装importlib-metadata库才能使用
- 高度可定制性、简单易用、图表交互性强，支持导出 SVG 格式图像

### Pandas

#### 定义

基于numpy数组构建的数据分析包

#### 数据结构

1. 一维数据结构Series：

   - 由一组数据和相关的标签（索引）组成

   - ```python
     import numpy as np
     import pandas as pd
     pd.Series(data,index=[])
     ```

   - 按索引名切片操作时，是包含终止索引的

   - Series在维度不匹配时index取并集，不能对齐的部分当缺失值NaN

2. 二维数据结构DataFrame

   - 表格型，既有行索引index也有列索引column

   - ```python
     import pandas as pd
     # 创建字典 d，包含两个 Series
     d = {
         'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
         'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])
     }
     
     # 如果没有传入 columns 的值，那么 columns 的值默认为字典 key，
     # index 默认为所有 value 中 index 的并集（自动对齐）
     df = pd.DataFrame(d)
     # 显示 DataFrame
     print(df)
     # 输出结果
        one  two
     a  1.0  1.0
     b  2.0  2.0
     c  3.0  3.0
     d  NaN  4.0
     ```

3. 三维数据结构Panel

#### I/O操作

- 读写csv文件
  - 写入：to_csv
  - 读取：pd.read_csv
- 读写Excel文件
  - 写入：to_excel
  - 读取：pd.read_excel

### Seaborn

| 特性       | Matplotlib                             | Seaborn                                |
| ---------- | -------------------------------------- | -------------------------------------- |
| 定位       | 基础绘图库，高度灵活，底层控制         | 高级统计绘图库，基于 Matplotlib 封装   |
| 代码复杂度 | 高（绘制复杂统计图需大量代码）         | 低（一行代码即可实现复杂统计图）       |
| 默认美观度 | 一般，需手动美化                       | 高，内置专业主题和调色板               |
| 数据输入   | 主要是数组/列表                        | 原生支持 Pandas DataFrame              |
| 统计功能   | 需手动计算统计量（如回归线、置信区间） | 自动计算并绘制（回归、分布、置信区间） |
| 多变量分析 | 需手动循环构建子图                     | 内置 `pairplot`, `FacetGrid` 轻松实现  |



