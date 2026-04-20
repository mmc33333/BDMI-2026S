# Mi0228学习记录

## Markdown使用：

### 1. LaTeX语法输入数学公式
$$
E = mc^2 + \cos(\theta)
$$

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### 2. 插入图片
![sample picture](./sample.jpeg)

### 3. 表格/流程图
| 模块 | 掌握内容 | 备注 |
|:-----|:--------:|-----:|
| Markdown | LaTeX公式、表格、流程图 | 练习 |
| Git | Gitee注册、基本指令 | init/add/commit |
| Python | 计算Pi的三种方法 | math库/级数/蒙特卡洛 |

---

## Git使用：

### 1. Gitee注册
已完成

### 2. Git指令复习
- git clone "https://gitee.com/mi729/BDMI-2026S"
- git add .
- git commit -m "update"
- git push

---

## Python3：

### 1. 基础数据类型
- number: 整数、浮点数、复数
- string: 字符串操作
- tuple: 不可变序列
- list: 列表（推导式、insert、pop、sort、random）
- set: 集合
- dictionary: 字典

### 2. 基本语句
- 条件判断：if-elif-else
- 循环：for、while

### 3. 函数定义
```python
def function_name():
    pass

if __name__ == '__main__':
    # 主程序入口
    pass

# lambda函数示例
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
```

### 4. 文件操作
#### 基础打开/关闭
```python
file = open(file_path, 'r')
file.close()
```

#### 路径操作
```python
import os
current_file = os.path.realpath('file_io.ipynb')
current_dir = os.path.dirname(current_file)
parent_dir = os.path.join(os.path.dirname(current_file))
```

#### with语句（自动关闭）
```python
with open(file_path, 'r') as simple_file:
    content = simple_file.read()
```

#### 写入文件
```python
with open(file_path, 'w') as f:
    f.write('content')
```

#### 删除文件
```python
import os
if os.path.exists(file_path):
    os.remove(file_path)
```

### 5. 类与面向对象
```python
class MyClass:
    def __init__(self, param):
        self.param = param
    
    def method(self):
        pass

class ChildClass(MyClass):
    def __init__(self, param, extra):
        super().__init__(param)
        self.extra = extra

class CallableClass:
    def __call__(self):
        print("实例被调用了")

obj = CallableClass()
obj()
```