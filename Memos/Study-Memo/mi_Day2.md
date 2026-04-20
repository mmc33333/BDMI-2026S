# Mi0303学习记录

## Numpy Practice

### 生成数组
```python
import numpy as np

a = np.array([1,2,3,4])
b = np.array(((1,2,3),[2,3,4],(3,4,5)))
c = np.array([[1,2,3],[4,5,6]],dtype=complex)

d = np.arrange(10)
e = np.arrange(4,8,2) #step

f = np.ones((3,3))
g = np.zeros((3,3))
h = np.identity(3)
```

### 基本运算&操作
```python
#+ - *
a @ b #dot


a.T #transpose
a.flatten() #降至一维
a.reshape(2,2)
a.shape
a.dtype
```

### 随机数
```python
import numpy as np
import numpy.random as npr

#生成x1到x2的均匀分布
a = npr.rand(10)*(x2 - x1) + x1
b = npr.rand(5,5)*(x2 - x1) +x1

#生成正态分布
npr.standard_normal(sample_size)
npr.normal(100,20,sample_size)

#生成卡方分布
npr.chisquare(df=0.5, size=sample_size)

#生成泊松分布
npr.poisson(lam=1.0, size=sample_size)
```
---
## Matplotlib Practice

### Scatter plot
```python
import matplotlib.pyplot as plt

plt.axis([0,5,0,20])
plt.title('My Scatter Plot')
plt.plot([1,2,3,4],[1,4,9,16],'ro')
```
### Line plot
```python
import math
import numpy as np
import matplotlib.pyplot as plt

t = np.arrange(0,2.5,0,1)
y1 = np.sin(math.pi*t)
y2 = np.sin(math.pi*t+math.pi/2)
y3 = np.sin(math.pi*t-math.pi/2)
plt.plot(t,y1,'b*',t,y2,'g^',t,y3,'ys')

#Subplot
plt.subplot(2,1,1)
...
plt.subplot(2,1,2)
...
```
### Bar plot
```python
import pygal

hist = pygal.Bar()
hist.title = ''
hist.x_labels = ['1','2','3']
hist.x_title = 'Result'
hist.y_title = 'Frequency'
hist.add('D6',frequencies) #frequencies数组已生成完毕
hist.render_to_file('Desktop/frequency.svg')
```

---
## Pandas Practice

### 数据结构
#### Series
```python
import pandas as pd

pd.Series(data,index=[])
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
#索引切片
s['b':'d'] s[1:3] #只有数字切片左闭右开
```
#### DataFrame
```python
pd.DataFrame(data, columns=[], index=[])

dates = pd.date_range('20200101', periods=6) #为金融数据而生
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

#也可从dict构建

#attributes: shape, index, ndims, dtypes, values, columns

#列操作
df['three'] = df['one'] + df['two']
df['flag'] = df['three'] > 0

```

#### Panel

pass

### 读写操作
```python
df.to_csv('BDMI.csv')
df.to_csv(r'地址')
df = pd.read_csv('BDMI.csv')
df = pd.read_csv(r'--')
#同理可以读写excel
```
---
## Seaborn Practice
```python
import seaborn as sns
import pandas as pd

df = pd.read_csv('~')
...
```
### displot
```python
sns.displot(df, x='Glucose') #得到关于Glucose的直方图 == sns.histplot(df,x='Glucose')
sns.displot(df, x='Glucose', hue='Outcome') #用不同颜色区分 增加col 分开表示
sns.displot(df, x='Glucose'，kind=‘kde') #绘制概率密度图 == sns.kdeplot()  也可以用kde=True
#kind='ecdf' 绘制概率累积图

#多变量绘图
sns.displot(df, x='Glucose', y='BMI', hue='Outcome')
```
### relplot
```python
sns.relplot(data=df, x='', y='') #直接绘制散点图
sns.relplot(data=df, x='', y='', hue='', style='') #对比散点图
#加入col 分裂关于col的多个图 

#jointplot pairplot
```
### catplot
sns.catplot(data=df, x='', jutter=False, y='') #直接绘制stripplot 分类型数据

kind = strip/box/boxen/swarm/violin