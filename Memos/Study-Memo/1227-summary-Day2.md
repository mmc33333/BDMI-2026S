## 20260303 BDMI课程小记
#### by 物理42 俞善斌
第二次课。
* 全面仔细地学习了有关**gitee**和**git**的操作。
* 介绍了**python**相关的程序工具，包括**jupyter**，**miniconda**，**VScode**等
* 介绍了**python**的常用库及其安装，包括**Numpy**，**Scipy**，**Matplotlib**，**Pandas**等，并对其中的一些库进行了进一步讲解。以下为各个库的介绍内容。
---
### Numpy
课程详细介绍了利用**Numpy**库进行的**随机数生成**相关的函数及其应用。  
NumPy是Python科学计算的基础库，提供了高效的多维数组对象和丰富的数学函数。

#### 1. 安装NumPy
在命令行中执行以下命令即可安装：
```bash
pip install numpy
```

#### 2. 创建数组
| 函数 | 描述 |
|------|------|
| `np.arange(start, stop, step)` | 定义范围（开始，停止，步长） |
| `np.ones(shape)` | 创建指定形状的全1数组 |
| `np.zeros(shape)` | 创建零矩阵 |
| `np.identity(n)` | 创建一个单位矩阵 |
| `np.random.random(size)` | 生成[0.0, 1.0)区间的随机浮点数数组 |

#### 3. 随机数生成
| 函数 | 描述 |
|------|------|
| `np.random.randn(d0, d1, ..., dn)` | 返回一个或一组随机数，具有标准正态分布 |
| `np.random.randint(low, high, size=None, dtype)` | 返回随机整数，范围区间为[low,high)；若high省略，默认生成范围[0, low) |
| `np.random.choice(a, size, replace=True, p=None)` | 从给定的一维数组中生成随机数，可指定概率p |
| `np.random.seed(seed)` | 使得随机数据可预测，设置相同的seed每次生成相同随机数 |

#### 4. 数组属性与变形
| 属性/方法 | 描述 |
|-----------|------|
| `array.shape` | 获取numpy数组的形状，也可直接赋值修改形状 |
| `array.dtype` | 检查数组的数据类型 |
| `array.reshape(new_shape)` | 转换数组的形状，返回新的数据对象 |
| `array.ravel()` | 将多维数组转换为一维数组 |
| `array.transpose()` 或 `array.T` | 调换数组的行列值的索引值，相当于转置 |
| `np.vstack((a, b))` | 垂直叠加2个数组 |
| `np.hstack((a, b))` | 水平叠加2个数组 |

#### 5. 算术运算与矩阵乘法
| 操作/函数 | 描述 |
|-----------|------|
| `a + 4` | 数组与标量的逐元素运算 |
| `a + b` | 数组与数组的逐元素运算 |
| `np.sin(arr)` | 对数组逐元素求正弦 |
| `np.sqrt(arr)` | 对数组逐元素求平方根 |
| `a * b` | 逐元素乘法（对应元素相乘） |
| `np.dot(A, B)` | 矩阵乘法（点积） |
| `np.matmul(A, B)` | 矩阵相乘 |

#### 6. 增减运算符
- Python中没有`++`和`--`，使用`+=`和`-=`实现自增自减。

#### 7. 调试相关
| 方法 | 描述 |
|------|------|
| `type(stuff)` | 获取变量的类型 |
| `print(f'My name is {name}')` | 输出信息，构造消息的简便方法 |
| `import pdb; pdb.set_trace()` | 设置断点进行调试 |
---
### Matplotlib
详细介绍了利用**Matplotlib**库进行各类图表的绘制的方法。同时，介绍了**pygal**库，用于制作更为精美的图表，有利于可视化展示。

#### 1. 安装Matplotlib
在命令行中执行以下命令即可安装：
```bash
pip install matplotlib
```

#### 2. 基本绘图函数
| 函数 | 描述 |
|------|------|
| `plt.plot(x, y, fmt)` | 绘制线图或散点图（通过fmt指定颜色和标记样式） |
| `plt.scatter(x, y, s, c, marker, alpha)` | 绘制散点图，可设置点大小、颜色、标记形状和透明度 |
| `plt.bar(x, height, width, color, alpha, label, yerr)` | 绘制垂直条形图，可设置误差线、透明度等 |
| `plt.barh(y, width, height, color, alpha, label)` | 绘制水平条形图 |
| `plt.hist(x, bins, color, alpha)` | 绘制直方图，bins指定分组数 |
| `plt.pie(x, labels, colors, explode, shadow, autopct, startangle)` | 绘制饼图，可设置突出显示、阴影、百分比标签、起始角度 |
| `plt.boxplot(x)` | 绘制箱线图（结合pandas的`df.boxplot()`使用） |
| `plt.contour(X, Y, Z, levels, colors, cmap)` | 绘制等高线图，可指定颜色或颜色映射 |
| `plt.contourf(X, Y, Z, levels, cmap)` | 绘制填充等高线图 |
| `plt.colorbar()` | 添加颜色条（用于等高线图或图像） |

#### 3. 图形定制函数
| 函数 | 描述 |
|------|------|
| `plt.title(label, fontsize, fontname)` | 设置图表标题，可指定字体大小和字体名称 |
| `plt.xlabel(label, color)` | 设置x轴标签，可指定颜色 |
| `plt.ylabel(label, color)` | 设置y轴标签，可指定颜色 |
| `plt.text(x, y, text)` | 在指定坐标添加文本注释 |
| `plt.grid(visible)` | 显示或隐藏网格线 |
| `plt.legend(labels, loc)` | 显示图例，loc指定位置（如'upper right', 'lower left'等） |
| `plt.xticks(ticks, labels)` | 设置x轴刻度位置和标签 |
| `plt.yticks(ticks, labels)` | 设置y轴刻度位置和标签 |
| `plt.axis([xmin, xmax, ymin, ymax])` | 设置坐标轴范围 |
| `plt.xlim(left, right)` | 设置x轴范围 |
| `plt.ylim(bottom, top)` | 设置y轴范围 |
| `plt.tight_layout()` | 自动调整子图参数，使之填充整个图像区域 |

#### 4. 子图与布局
| 函数 | 描述 |
|------|------|
| `plt.subplot(nrows, ncols, index)` | 将当前图像划分为多个子图，并选择当前子图 |
| `plt.figure(figsize)` | 创建新图像，可指定图形大小 |
| `fig.add_axes([left, bottom, width, height])` | 在图形上添加坐标轴（用于创建多个独立坐标系） |

#### 5. 图表保存与显示
| 函数 | 描述 |
|------|------|
| `plt.savefig(filename)` | 将当前图表保存为图像文件（如PNG） |
| `plt.show()` | 显示图表 |

#### 6. 3D绘图（需导入`mpl_toolkits.mplot3d`）
| 函数 | 描述 |
|------|------|
| `from mpl_toolkits.mplot3d import Axes3D` | 导入3D工具 |
| `ax = Axes3D(fig)` | 创建3D坐标轴对象 |
| `ax.plot_surface(X, Y, Z, cmap)` | 绘制3D曲面，可指定颜色映射 |
| `ax.scatter(xs, ys, zs, c, marker)` | 绘制3D散点图 |
| `ax.set_xlabel(label)`, `ax.set_ylabel(label)`, `ax.set_zlabel(label)` | 设置3D坐标轴标签 |
| `ax.view_init(elev, azim)` | 设置3D视图的仰角和方位角 |
---
### Pandas
详细介绍了利用**Pandas**库进行数据处理管理的方法。Pandas提供了高级数据结构和数据操作工具，适用于数据清洗、分析和挖掘。

#### 1. 安装与导入
| 命令/代码 | 描述 |
|-----------|------|
| `conda install -c conda-forge pandas` | 使用conda安装Pandas（也可用`pip install pandas`） |
| `import pandas as pd` | 导入Pandas库，通常简写为pd |

#### 2. Series（一维数据结构）
| 函数/操作 | 描述 |
|-----------|------|
| `pd.Series(data, index=index)` | 创建一维Series，data可为列表、数组等，index指定索引（默认0,1,2...） |
| `s[index]` | 通过索引选取单个或一组值 |
| `s + s` | 向量化操作，索引对齐后进行逐元素运算 |
| `s * 2` | 标量运算，逐元素乘以2 |
| `s[1:] + s[:-1]` | 索引不对齐时，结果取索引并集，缺失部分为NaN |

**注意**：按索引名切片时，包含终止索引。

#### 3. DataFrame（二维数据结构）
| 函数/方法 | 描述 |
|-----------|------|
| `pd.DataFrame(data, columns=columns, index=index)` | 创建二维DataFrame，data可为二维数组、Series字典等 |
| `pd.date_range(start, periods, freq)` | 生成日期时间索引，例如`pd.date_range('20200101', periods=6)` |
| `df.index` | 获取行索引 |
| `df.columns` | 获取列索引 |
| `df.shape` | 获取DataFrame的行数和列数 |
| `df.dtypes` | 获取各列的数据类型 |
| `df.ndim` | 获取数据维度（2表示二维） |
| `df.values` | 获取对象值，返回二维ndarray数组 |

#### 4. DataFrame列操作
| 操作 | 描述 |
|------|------|
| `df[col]` | 选择单列，返回Series |
| `df[new_col] = ...` | 添加新列，可赋值为标量、列表或Series |
| `df['three'] = df['one'] * df['two']` | 通过已有列运算创建新列 |
| `df['flag'] = df['one'] > 2` | 布尔运算创建新列（值为True/False） |

#### 5. 文件读写（I/O）
| 函数 | 描述 |
|------|------|
| `df.to_csv('filename')` | 将DataFrame写入CSV文件 |
| `pd.read_csv('filename')` | 从CSV文件读取数据到DataFrame |
| `df.to_excel('filename')` | 将DataFrame写入Excel文件 |
| `pd.read_excel('filename')` | 从Excel文件读取数据到DataFrame |

#### 6. 其他常用方法
| 方法 | 描述 |
|------|------|
| `df.head()` | 查看DataFrame的前几行（默认前5行） |
| `df.describe()` | 生成描述性统计信息（如计数、均值、标准差等） |
---
### Seaborn
在已经介绍了Matplotlib的基础上，详细介绍了**seaborn**库。这是一个基于Matplotlib的绘图库，对许多功能进行了更好的封装，使得制作美观、信息丰富的统计图形变得更加容易。

#### 1. 安装与导入
| 命令/代码 | 描述 |
|-----------|------|
| `pip install seaborn` | 使用pip安装seaborn |
| `import seaborn as sns` | 导入seaborn库，通常简写为sns |

#### 2. 分布图 (Distribution plots)
用于展示数据分布情况的图表。

| 函数 | 描述 |
|------|------|
| `sns.histplot(data, x, bins, kde)` | 绘制直方图，可选叠加核密度估计曲线 |
| `sns.kdeplot(data, x, shade)` | 绘制核密度估计曲线，可填充阴影 |
| `sns.ecdfplot(data, x)` | 绘制经验累积分布函数图 |
| `sns.displot(data, x, kind='hist')` | 高级接口，通过`kind`参数选择分布图类型（'hist', 'kde', 'ecdf'），支持分面绘图 |

#### 3. 关系图 (Relational plots)
用于展示两个或多个变量之间关系的图表。

| 函数 | 描述 |
|------|------|
| `sns.scatterplot(data, x, y, hue, size, style)` | 绘制散点图，可通过颜色、大小、样式映射更多变量 |
| `sns.lineplot(data, x, y, hue, style, markers)` | 绘制线图，常用于时间序列或连续数据 |
| `sns.relplot(data, x, y, kind='scatter', col, row)` | 高级接口，通过`kind`参数选择关系图类型（'scatter'或'line'），支持分面绘图 |

#### 4. 分类图 (Categorical plots)
用于展示分类变量与数值变量之间关系的图表。

| 函数 | 描述 |
|------|------|
| `sns.boxplot(data, x, y, hue)` | 绘制箱线图，展示数据分布的五数概括及异常值 |
| `sns.violinplot(data, x, y, hue, split)` | 绘制小提琴图，结合箱线图和核密度估计 |
| `sns.barplot(data, x, y, hue, ci)` | 绘制条形图，默认显示均值及置信区间 |
| `sns.pointplot(data, x, y, hue, ci, join)` | 绘制点图，显示点估计及置信区间，常与线连接 |
| `sns.catplot(data, x, y, kind='box', col, row)` | 高级接口，通过`kind`参数选择分类图类型（'box', 'violin', 'bar', 'point'等），支持分面绘图 |

#### 5. 其他常用图表
| 函数 | 描述 |
|------|------|
| `sns.heatmap(data, annot, fmt, cmap)` | 绘制热力图，常用于展示相关性矩阵或矩阵数据 |
| `sns.pairplot(data, hue, vars)` | 绘制配对图，展示数据集中所有数值变量两两之间的关系 |
| `sns.jointplot(data, x, y, kind='scatter')` | 绘制联合分布图，同时展示单变量分布和双变量关系 |
| `sns.clustermap(data, cmap)` | 绘制聚类热力图，对行和列进行层次聚类 |

#### 6. 主题与样式设置
| 函数 | 描述 |
|------|------|
| `sns.set_theme(style='whitegrid')` | 设置全局绘图主题，可选'white', 'dark', 'whitegrid', 'darkgrid', 'ticks' |
| `sns.set_palette(palette)` | 设置颜色调色板，如'hls', 'husl', 'Set2'等 |
| `sns.despine()` | 移除上轴和右轴的边框线，使图形更简洁 |