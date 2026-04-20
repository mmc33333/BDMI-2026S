# 1.课堂热身:发送弹幕
# 2.一些讲解
课程将于12周左右结束，每周上四学时。
课程路线图：
![课程路线图](<images/Pasted image 20260303134300.png>)
课程需要每周提交总结文档
打开雨课堂和md记录工具
# 3.git+gitee演示

![流程](<images/Pasted image 20260303134810.png>)

- git add:暂存你所做的修改
- git commit：将所作的修改进行标记，表明所做的修改已经完成
- git push：更新推送至远程仓库
- 在远程仓库创建pull request，更新至源仓库
- 审查更新，检查代码是否有冲突。
# 4.计算平台的使用
jupyter notebook
# 5.py setup
## 1.下载pandas
创建虚拟环境，推荐使用miniconda
## 2.下载vscode
确定python环境

# 6.numpy库
- 下载：使用cmd
pip install
## 1.使用array创建数组
![example](<images/Pasted image 20260303141205.png>)
## 2.函数运算
## 3.矩阵乘积
代数相乘：使用dot（）函数：矩阵乘法
相乘：A* B：逐分量相乘
## 4.增减算符
+= ，-= 
## 5.数组变形
使用reshape函数：更换数组形状
ravel：多维数组变为一维数组
transpose：转置
## 6.随机数
- numpy.random
- 均匀分布的随机数
npr.rand(10):[0,1)随机分布数列

 - 正态分布的随机数
 ```python
 npr.standard_normal(sample_size)标准正态分布
 npr.normal(100,20,sample_size)正态分布

 ```
 

 - 卡方分布
 ```python
 npr.chisquare(df=0.5,size=sample_size)
 
 ```

   - 泊松分布
   ```python
   npr.poison(lam=1.0 size=sample_size)
   ```

## 7.蒙特卡洛模拟
![example1](<images/Pasted image 20260303143447.png>)
![conclusion](<images/Pasted image 20260303143654.png>)
# 7.matplotlib画图
import matplotlib.pyplot as plt
- 绘制散点图
- 绘制线图
- 一图多线，更换不同颜色
- subplot()绘制子图

# 8.pygal画图工具
- python画廊
- 可视化工具
- 模拟仿真simulation
主要是生成随机数并进行计数

# 9.matprolib高级用法
- 使用多个图形和轴
- text（），title（）
- 添加grid和legend
函数为添加图例，默认添加到右上角
- 将图标另存为图像
```python
savefig()
```
- 处理日期值
- line charts
- histograms（直方图）
```python
hist()
```
- bar charts
```python
	bar()
```
使用kawargs标记误差线
水平条形图：barh
其他各种用法...

- pie chart(饼图)
- 等值线图contour plot
- 高级图表contour plot
- mplot3d toolkit
绘制3D表面图
- subplot嵌套subplot

# 10.pandas
- 基于numpy开发
![intro](<images/Pasted image 20260303151043.png>)
### 1.安装pandas
### 2.介绍
![intro](<images/Pasted image 20260303151303.png>)
- 一维数据结构
```python
pd.Series(data,index=[])
```
series:一维数据结构，由一组数据和一组与之相关的数据标签（索引）构成：index value
可以通过索引的方式选取一个或者一组数据
```python
# 语法：pd.Series(data, index=[ ])
# 若忽略 index，则索引默认为 0, 1, 2, 3, ..., len(data)-1

s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
```
索引操作：
```python
# 1. 行索引 (Row index)
print(s['b'])
print(s[2])

# 2. 切片索引 (Slice index)
# 注意：在 Pandas 中按“索引名”切片操作时，是包含终止索引的！
print(s['b':'d']) 

# 3. 不连续索引 (Discontinuous index)
print(s[[0, 2, 4]])
print(s[['a', 'e']])

```
向量化操作：与ndarray表现一致
标签对齐：Series 之间操作时，默认会使用 index 的值进行对齐，而不是相对位置
```python
# 如果两个 Series 不能完全对齐，结果的 index 是两者的并集
# 不能对齐的部分会被当作缺失值 (NaN) 处理
s[1:] + s[:-1]
```
- 二维数据结构
既有行索引，也有列索引
```python
# 语法：pd.DataFrame(data, columns=[ ], index=[ ])

# 方法一：通过二维数组构建 (DataFrame from 2-d array)
dates = pd.date_range('20200101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

# 方法二：从 Series 字典中构造 (DataFrame from dict of Series)
d = {
    'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
    'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])
}
# 如果没有传入 columns 的值，默认将字典的 key 作为 columns，index 为所有 value 中 index 的并集
df = pd.DataFrame(d)
```
![result](<images/Pasted image 20260303152622.png>)
dataframe的基础属性
```python
df.shape    # 查看维度（行数, 列数）
df.dtypes   # 查看各列的数据类型
df.ndim     # 查看数据维度
df.index    # 查看行索引
df.columns  # 查看列索引
df.values   # 查看对象值（返回二维 ndarray 数组）
```
如果指定了Column的值，会在字典中寻找，找不到的是nan
- 列操作
```python
# 1. 选取列 (Selection)
df['one']

# 2. 列的计算与新增 (Addition)
df['three'] = df['one'] * df['two']

# 3. 布尔运算列 (Bool operation)
df['flag'] = df['one'] > 2
```
- 读写文件
#### 1.读写csv文件
```python
# 读取 CSV 文件
df_2 = pd.read_csv('BDMI.csv')
df_4 = pd.read_csv(r'C:\course\test.csv')

# 保存写入 CSV 文件
df.to_csv('BDMI.csv')
df_3.to_csv(r'C:\course\mydata.csv')
```
#### 2.读写excel文件
```python
# 读取 Excel 文件
df_2 = pd.read_excel('findmydata.xlsx')
df_4 = pd.read_excel(r'C:\course\mydata.xlsx')

# 保存写入 Excel 文件
df_3.to_excel('findmydata.xlsx')
df_3.to_excel(r'C:\course\mydata.xlsx')
```
- 使用智谱清言ai进行辅助编程

# 11.seaborn可视化绘图
基于matplotlib统计数据的图形可视化工具
- 更高级的封装
- matplotlib更灵活
![intro](<images/Pasted image 20260303153717.png>)
数据准备：使用数据集
### 1.displot作图
```python
sns.displot(df, x='Glucose')
sns.hisplot(df, x='Glucose')#两种直方图的画法
```
```python
sns.displot(df, x='Glucose', hue='Outcome',col='Outcome')#hue参数使用不同颜色加以区别，col将数据分列展示
```
displot-kdeplot
```python
sns.kdeplot(data=df, x='Glucose', hue='Outcome',col='Outcome')
sns.displot(df, x='Glucose', kind='kde',col='Outcome')
```
- 生成概率图
displot-ecdf
![example](<images/Pasted image 20260303154653.png>)
- displot多变量
![example](<images/Pasted image 20260303154736.png>)
### 2.relplot
- relplot-scatterplot
![1](<images/Pasted image 20260303154929.png>)
style参数根据类别使用不同形状，推荐和hue使用相同变量
![2](<images/Pasted image 20260303154954.png>)

使用col或row绘制子图区分新的种类
![3](<images/Pasted image 20260303155054.png>)
- replot-lineplot
![4](<images/Pasted image 20260303155139.png>)
- jointplot&pairplot
![1](<images/Pasted image 20260303155258.png>)
![2](<images/Pasted image 20260303155326.png>)
将数据集的每一列全部绘制出来

- catplot-scatter
![example](<images/Pasted image 20260303155507.png>)
![example](<images/Pasted image 20260303155607.png>)
- catplot-distribute
根据分位数分箱，box四分位数，boxen更多分位
![eg](<images/Pasted image 20260303155807.png>)
参数：
![参数](<images/Pasted image 20260303155855.png>)

















