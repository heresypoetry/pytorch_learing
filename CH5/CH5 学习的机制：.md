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

注：上诉代码返回的grad就是损失函数以w和b为自变量，得到的梯度。

---

有了上诉梯度，我们可以对w,b进行不断更新，直到，w,b趋于稳定，得到的w,b可以使得梯度取到接近最小值。

下面定义循环函数：

```Python
def training_loop(n_epochs, learning_rate, params, tu, tc):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        tp = model(tu, w, b)  
        Loss = loss(tp, tc)
        Grad = grad(tu, tc, tp, w, b)  
        params = params - learning_rate * Grad
        #print('Epoch %d, Loss %f' % (epoch, float(Loss)))
        #print('Params:', params)
        #print('Grad:', Grad) 
        #上面注释的这几行会导致可能会有很长的输出
    return params
```

然后运行：

```python
training_loop(
    n_epochs = 100, 
    learning_rate = 1e-2, 
    params = torch.tensor([1.0, 0.0]), #初始值
    tu = tu, 
    tc = tc)
```

- 过度训练：

上面的例子中，我们发现训练过程完全崩溃了，损失越来越大，反而变成了无穷。这并不是程序出错了，而是因为参数收到的更新太大了，它的值来回波动，每次更新都修正过度，导致下一次更新过度。所以它发散到无穷而不是收敛。此时可以选择降低学习率。当学习率足够低，则最后会收敛。但是学习率太低会使得学习的速度很慢。可以手动调整学习率（这种手动调整模型中的不参与训练的参数的过程叫做超参数学习）。可以让学习率根据结果变化自动调整，这叫做学习率自适应。

最后函数值收敛，w,b分别应该是 0.2327, -0.0438。



- 归一化输入：

在上面的例子中。第一百次迭代得到的梯度为（-0.0532,  3.0226）可见二者数量级差了几十倍。但是两个参数的学习率相同。实际上两个参数所需要的能够快速收敛的学习率是不一样的。如果学习率足够大，会出现能有效更新一个参数，另一个参数不稳定的情况。我们可以给不同参数不同的学习率。但当参数多时，这太麻烦了。所以我们需要归一化输入，使得输入的范围不会偏离-1.0～1.0太远。

比如，`tu_n = tu*0.1` , tu_n表示归一化之后的输入。

修改training_loop函数：

```Python
tu_n = tu*0.1
params = training_loop(
    n_epochs = 100, 
    learning_rate = 1e-2, 
    params = torch.tensor([1.0, 0.0]), 
    tu = tu_n, 
    tc = tc)
```

这次，即使使用1e-2的学习率，结果也成功收敛。最后梯度是（-0.4446,  2.5165），二者在数量级上接近，说明不需要再归一化。

考虑更多次迭代，发现5000次之后左右，数据趋于稳定。最后得到的向量是：（5.3671, -17.3012）。由于之前归一化的过程，w = 0.1* 5.3671。

回顾之前温度计的读数，tc = 0.5367*tu -17.3012.

举例计算发现：35.7*0.5367-17.3012 = 1.9 , 0.5367 * 55.9 -17.3012 = 12.7. 

68. 4* tu -17.3012 = 19.4.

模型结果与运算结果差距还是比较小的。

----

- 可视化数据：

```python
from matplotlib import pyplot as plt
fig = plt.figure(dpi = 600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(tu.numpy(), tp.detach().numpy()) 
plt.plot(tu.numpy(), tc.numpy(), 'o')
plt.savefig("temp_unknown_plot.png", format="png") 
```

关于的使用，可以参考这个回答：

{https://www.zhihu.com/question/588580149/answer/2959008082}

----

- Pytorch 自动求导：

PyTorch 张量可以记住它们自己从何而来，根据产生它们的操作和父张量，它们可以根据输人自动提供这些操作的导数链。这意味着我们不需要手动推导模型，给定一个前向表达式，无论嵌套方式如何，PyTorch 都会自动提供表达式相对其输人参数的梯度。所有的张量都有一个grad属性，默认情况下，该属性对应的值是None. 如果在构造张量的时候，设置构造函数的requires_grad-True参数，这个参数告诉PyTorch跟踪由对params张量进行操作后产生的张量的整个系谱树。任何将params作为祖先的张量都可以访问params到那个张量调用的函数链。

图见教材 p109

我们可以有任意数量的 requires-grad 为 True 的张量和任意组合的函数。在这种情况下， PyTorch 将计算整个函数链（计算图）中损失的导数，并将它们的值累加到这些张量的 grad属性中（图的叶节点)。

注意，这是许多 PyTorch 初学者以及一些有经验的人经常会弄错的，我们在这里要强调的是**导数是累加存储到 grad 属性中的**。

即，调用backward()将导致导数在叶节点相加。因此如果提前调用backward(),则会再次计算损失。再次调用backword()，每个叶节点上的梯度将在上一次迭代的基础上累加，导致梯度计算错误。所以在每次迭代之前都要将梯度归零。使用zero()_方法。_

```python
if params.grad is not None:
	params.grad.zero_()
```

这一步是必要的。

接下来是本节例子的完整算法：

```Python
import torch
import numpy as np
tc = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] 
tu = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8,48.4, 60.4, 68.4]
def model(tu,w,b): 
    return w * tu+b
# 定义了一个模型
tc = torch.tensor(tc)
tu = torch.tensor(tu)
tu_n = 0.1*tu
#别忘了转化为张量
def loss(tp,tc):
    squared_diffs = (tp-tc)**2
    return squared_diffs.mean()
params = torch.tensor([1.0,0.0],requires_grad = True)
def training_loop(n_epochs,learning_rate,params,tu,tc):
    for epoch in range(1,n_epochs+1):
        if params.grad is not None:
            params.grad.zero_()
        tp = model(tu,*params)
        Loss = loss(model(tu,*params),tc)
        Loss.backward() 
        with torch.no_grad():
            params -= learning_rate*params.grad
        if epoch % 200 == 0:
            print('epoch %d:%f' %(epoch,float(Loss)))
    return params
training_loop(5000,1e-2,params,tu_n,tc)
```

---

- 优化器：
- 通过使用优化器，可以让我们的代码更加简单。

torch模块有一个字模块，叫optim。里面包含了实现不同优化算法的类。

每个优化器构造函数接受一个张量（其中requires_grad设置为True）作为第一个输入，传递给优化器的所有参数都保留在优化器对象中，这样优化器可以更新并访问他们的值。优化器可以减少很多的代码量。比如之前我们写的梯度下降的方法，可以使用叫SGD（随机梯度下降，随机的意思是这个算法只利用所有输入样本的一个随机子集得到的，而我们的算法利用的是所有子集）

```Python
import torch
import numpy as np
import torch.optim as optim
tc = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] 
tu = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8,48.4, 60.4, 68.4]
def model(tu,w,b): 
    return w * tu+b
# 定义了一个模型
tc = torch.tensor(tc)
tu = torch.tensor(tu)
tu_n = 0.1*tu
#别忘了转化为张量
def loss(tp,tc):
    squared_diffs = (tp-tc)**2
    return squared_diffs.mean()
params = torch.tensor([1.0,0.0],requires_grad = True)
learning_rate = 1e-2
optimizer = optim.SGD( [params], learning_rate )
def training_loop(n_epochs,optimizer,params,tu,tc):
    for epoch in range(1,n_epochs+1):
        tp = model(tu,*params)
        Loss = loss(tp,tc)
        optimizer.zero_grad()#这个方法替换了前面手动判断再清零。
        Loss.backward() 
        optimizer.step()# 上面这个语句就完成了对params的一次优化，修改了优化器的值。
        if epoch % 200 == 0:
            print('epoch %d:%f' %(epoch,float(Loss)))
    return params
training_loop(5000,optimizer,params,tu_n,tc)
print(params)
```

- 分割训练集：

randperm()函数可以把一个张量的元素打乱。容后从打乱的元素中挑一些作为训练集，一些作为测试集。