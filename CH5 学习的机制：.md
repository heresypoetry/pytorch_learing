# CH5 学习的机制：

- 损失函数：

用来表示训练样本的期望输出与输入这些样本时模型实际产生的输出之间的差值。越大说明差的越多。我们希望训练的模型可以使损失函数最小化。



- 在本节中，讨论了一个温度计的例子。有一个不显示单位的温度计，想要建立一个数据集，记录现在的温度和温度计的读数。然后设立一个模型，该模型可以由温度计的读数给出当前单位为摄氏度的温度。



以tc表示以摄氏度为单位的温度，tu使用的是我们未知的单位。我们在这里记录数据：

```python
import torch
import numpy as np
tc = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] 
tu = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8,48.4, 60.4, 68.4]
tc = torch.tensor(tc)
tu = torch.tensor(tu)
#别忘了转化为张量
```

根据常理以及对数据的观察，二者关系一般是线性的。

可以假设$tc = tu\times w + b.$

其中，w,b分别叫做权重和偏置。

用tp表示使用训练的模型得到的数据，对于损失函数，可以选择$|tp-tc|$, 或者$(tp-tc)^2$。在这里，我们选择后者。因为平方差比绝对差对错误结果的处罚更大（当错误结果误差超过1）。



现在使用PyTorch：

```python
def model(tu,w,b): 
    return tu*w+b
# 定义了一个模型
def loss(tp,tc):
    squared_diffs = (tp-tc)**2
    return squared_diffs.mean()
```

接着举个例子：

```Py
w = torch.ones(())
b = torch.zeros(())
tp = model(tu,w,b)
print(tp)
print(loss(tp,tc))
```



- 说明：本节中，w=1,b=0是零维张量。而列表转化而来的两个张量是1维的。其实，在很早期的PyTorch版本，只能对形状相同的参数使用基于元素的二元运算。在每个张量的匹配位置对于计算的结果。后来，PyTorch改写了在Numpy中流行的**广播机制**。对于基于元素的二元运算，其匹配机制如下：

​	由后向前迭代每个索引维度，如果其一个操作数的维度大小是1，那么 PyTorch 将使用该维度上的单个项与另一个张量沿该维度上的每一项进行运算。

​	如果两个维度大小都大于1，则它们的维度大小必须相同，并使用自然匹配。

​	如果一个张量的维度大于另一个张量的维度，那么另一个张量上的所有项将和这些维度上的每一项进行运算。例如教材上给了这样的例子：

```python
x = torch.ones(())
y = torch.ones(3,1)
z = torch.ones(1,3)
a = torch.ones(2, 1, 1)
print(f"shapes: x: {x.shape}, y: {y.shape}")
print(f"        z: {z.shape}, a: {a.shape}")
print("x * y:", (x * y).shape)
print("y * z:", (y * z).shape)
print("y * z * a:", (y * z * a).shape)
```

---

## 沿着梯度下降：

根据参数使用梯度下降法来优化损失函数。梯度下降思想是计算各参数的损失变化率，并在减小损失变化率的方向上修改各参数。，我们可以通过在w和b上加上一个小数字来估计变化率，然后看看损失在这附近的变化有多大。使用一个叫做学习率的比例因子来衡量w的值相对损失函数的变化的变化率。例如：

```python
delta = 0.1
learning_rate = 1e-2
loss_rate_of_change_w = (loss(model(tu, w + delta, b), tc) - loss(model(tu, w - delta, b), tc)) / (2.0 * delta)
w = w - learning_rate * loss_rate_of_change_w
loss_rate_of_change_b = (loss(model(tu, w, b + delta), tc) - loss(model(tu, w, b - delta), tc)) / (2.0 * delta)
b = b - learning_rate * loss_rate_of_change_b
```

通过观察w,b的变化手动调整变化率和dela的方法比较粗糙。

事实上，令delta无限小时，就是对损失函数的平均值求导，简单的计算得到，需要定义它的导数，

```Python
def dloss(tp, tc): return 2 * (tp - tc) / tp.size(0)
def dmodel_dw(tu,w,b): return tu
def dmodel_db(tu,w,b): return 1.0
#分别表示模型对w,对b求导。
def grad(tu,tc,tp,w,b):
    dloss_dtp = (dloss(tp,tc))
    dloss_dw = dloss_dtp*dmodel_dw(tu,w,b)
    dloss_db = dloss_dtp*dmodel_db(tu,w,b)
    return torch.stack([dloss_dw.sum(),dloss_db.sum()])
# stack是PyTorch的拼接函数。可以把多个k维的张量拼接为一个k+1维的张量。
```

