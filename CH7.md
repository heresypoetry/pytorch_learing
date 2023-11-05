# CH7 从图像学习：

本章以区分鸟和飞机为例，介绍图像学习。

## 微小图像数据集；

MNIST(手写数据识别)和CIFAR-10都是简单有趣，经典的数据集。本节以CIFAR-10为例。

CIFAR-10由60000张微小的（32像素$\times$32）RGB图像组成。用一个整数对应10个级别中的一个。分别是：

飞机（0）、汽车（1）、鸟（2）、猫（3）、鹿（4)、狗（5）、青蛙（6）、马（7）、船（8），卡车（9）

如今CIFAR-10被认为过于简单，无法应用于开发或者验证新的研究。

```python
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
data_path = '/Users/apple/Desktop/python'
cifar10 = datasets.CIFAR10(root= data_path,train = True, download=True)
cifar10_val = datasets.CIFAR10(root= data_path,train = False, download=True)
#download表示如果指定位置找不到，是不是要去网上下载。train指定我们对训练集还是验证集感兴趣。
print(len(cifar10))
```



- Dataset 类：

这是一个需要实现两种函数的对象. \_\_len()\_\_ ,和\_\_getitem\_\_. 前者返回数据中的项数，后者返回由样本以及它所对应的标签组成的项。

- Dataset变换：

通过`from torchvision import transforms`引入torchvision.transforms模块，这个模块定义了一组可组合的类似于函数的对象。例如，

```Python
from torchvision import transforms
to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
print(img_t.shape)
```

返回的结果是torch.size([3,32,32])。 对应CHW：通道，高度，宽度。

我们可以在定义数据集时就做transform，例如，我们可以将以前的代码修改为

```python
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
data_path = '/Users/apple/Desktop/python'
cifar10 = datasets.CIFAR10(root= data_path,train = True, download=True,transform= transforms.ToTensor)
cifar10_val = datasets.CIFAR10(root= data_path,train = False, transform= transforms.ToTensor)
#download表示如果指定位置找不到，是不是要去网上下载。train指定我们对训练集还是验证集感兴趣,把图片转化为了张量。
```

注：原始的PIL图像中每个通道由八位二进制表示（0～255）。 ToTensor变换将数据变换为每个通道的32位浮点数，将值缩小至0.0~1.0。

- 数据归一化：

通过选择在 $0 \pm1(or 2)$ 附近为线性的激活函数，将数据保持在相同的范围内意味着神经元更有可能具有非零梯度。因此可以更快地学习。

在之前的基础上用stack()函数将所有张量合并起来一起归一化（我们的数据量不大，可以这样做。）

```python
imgs = torch.stack([img_t for img_t,_ in cifar10],dim = 3)
print(imgs.shape)
```

返回torch.size([3, 32, 32, 50000])。

下面利用imgs计算三个通道上的平均值和标准差。

```python
print(imgs.view(3,-1).mean(dim = 1))
print(imgs.view(3,-1).std(dim = 1))
```

最后得到的结果是tensor([0.4914, 0.4822, 0.4465])和tensor([0.2470, 0.2435, 0.2616])

记住这两组数据（不然以后每次运行程序时都要重新算），将他们Normalize变化.重新修改数据的读入，在读入时就Normalize。 去掉stack，输出总体的均值，标准差。然后代码变为：

```python
data_path = '/Users/apple/Desktop/python'
cifar10 = datasets.CIFAR10(root= data_path,train = True, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))]))
cifar10_val = datasets.CIFAR10(root= data_path,train = False, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))]))
```





## 区分鸟和飞机：

我们在观鸟俱乐部的朋友 Jane，在机场南面的树林里布置了一组相机。当有东西进人镜头画面时，这些相机会拍摄并保存照片，将其上传到俱乐部的实时观乌博客上。问题是很多飞机进出机场都会触发摄像头拍照，所以 Jane 花了很多时间从博客上删除飞机照片。她需要的是一个自动化系统。她需要的不是人工删除，而是一个神经网络，依靠人工智能实现立刻自动剔除飞机的照片。

- 构建数据集：

本节利用之前cigar10的例子。因为cigar10的label为0，2的两类分别是飞机和鸟。在上诉预处理之后，直接引用。

在上一节代码之后，

```python
label_map = {0:0,2:1}
class_names = ['airplane','nird']
cifar2 = [(img,label_map[lable]) for img ,lable in cifar10 if lable in [0,2]]
cifar2 = [(img,label_map[lable]) for img ,lable in cifar10_val if lable in [0,2]]
#得到了新的数据集，使得飞机的标签是0，鸟的标签是1.注：这里cifar2是list
```

- 全连接模型：

我们的每个样本有$32\times 32 \times 3 = 3072$ 个特征，将这些放入一个线性模型（nn.Linear）， 

![CH7_1](/Users/apple/Library/Application Support/typora-user-images/ml_learn_CH7_1.png)



- 下面开始定义模型:

```python
import torch.nn as nn
n_out = 2 #模型最后输出只有两个特征
model = nn.Sequential(
    nn.Linear(3072,512),nn.Tanh(),nn.Linear(512,n_out)
)
#这是一个很简单的神经网络，只有两层。中间夹一个激活函数。
```

模型的输出有两个，根据他们的大小关系，可以分别将它理解为结果是飞机的概率和结果是鸟的概率。为了保证每对元素总和为1，还可以使用Softmax, 其表达式为
$$
Softmax(x_1,\dots,x_k) = (\frac{e^{x_1}}{\sum_{i=1}^{i=k}e^{x_i}},\dots,\frac{e^{x_k}}{\sum_{i=1}^{i=k}e^{x_i}})
$$


nn 模块将 Sofumax 作为一个可用的模块。nn.Softmax()要求我们指定用来编码的维度。现在我们在我们的模型后面加一个nn.Softmax().然后我们就得到了概率。

- 损失函数

我们有一些训练集，我们希望对于飞机来说，判断出是飞机的概率大于是鸟的概率。对于鸟来说，判断出是鸟的概率大于飞机的概率，而不关心具体概率是多少。所以，我们最希望惩罚的是错误分类的概率，而不是那些不完全像0或者不完全像1的东西。

我们考虑与之前的类相关的概率。我们希望的损失函数满足当概率很低的时候，损失非常高，当概率高于其他选择时，损失很低，而不是真的专注于将概率提升到1.

损失函数负对数似然（Negative Log Likelihood, NLL）可以满足我们的要求。

PyTorch有一个nn.NLLLoss类，它不是取概率而是取对数概率张量作为输入，原因是当输入的概率接近0时，取概率的对数是一件棘手的事，解决方法是用nn.Softmax()而不是nn.Softmax()，确保计算结果在数字上稳定。

这是因为，比如说$Softmax(x_1,x_2)$, 
$$
\ln(\frac{e^{x_1}}{e^{x_1}+e^{x_2}}) = x_1 - \ln(e^{x_1}+e^{x_2})
$$
$e^{x_1}+e^{x_2}$ 不会接近0.

于是可以修改Softmax() 为 LogSoftmax().

```python
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
from torch import optim
data_path = '/Users/apple/Desktop/python'
cifar10 = datasets.CIFAR10(root= data_path,train = True, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))]))
cifar10_val = datasets.CIFAR10(root= data_path,train = False, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))]))
label_map = {0:0,2:1}
class_names = ['airplane','nird']
cifar2 = [(img,label_map[lable]) for img ,lable in cifar10 if lable in [0,2]]
cifar2 = [(img,label_map[lable]) for img ,lable in cifar10_val if lable in [0,2]]

import torch.nn as nn
n_out = 2 #模型最后输出只有两个特征
model = nn.Sequential(
    nn.Linear(3072,512),nn.Tanh(),nn.Linear(512,n_out),nn.LogSoftmax(dim = 1)
)
loss_fn = nn.NLLLoss()
learning_rat = 1e-2
import torch.optim as optim
optimizer = optim.SGD(model.parameters(),learning_rat)
n_epochs = 100
for epoch in range(n_epochs):
    for img,label in cifar2:
        out = model(img.view(-1).unsqueeze(0))
        loss = loss_fn(out,torch.tensor([label]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d,Loss : %f" %(epoch,float(loss)))
#这是一个很简单的神经网络，只有两层。中间夹一个激活函数。
```

- 小批量运行：

通过上面的运行我们发现代码运行的太慢了。可能我们的数据量还是比较大。

torch.utils.data模块有一个DataLoader类。该类有助于打乱数据和组织数据。DataLoader构造函数至少接收一个数据集对象作为输入，以及batch_size（子数据集的大小）。返回的结果中为一些大小为batch_size$\times$原来张量的大小 的张量，之前的代码修改为：

```Python
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
from torch import optim
import torch.nn as nn
data_path = '/Users/apple/Desktop/python'
cifar10 = datasets.CIFAR10(root= data_path,train = True, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))]))
cifar10_val = datasets.CIFAR10(root= data_path,train = False, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))]))
label_map = {0:0,2:1}
class_names = ['airplane','nird']
cifar2 = [(img,label_map[lable]) for img ,lable in cifar10 if lable in [0,2]]
cifar2 = [(img,label_map[lable]) for img ,lable in cifar10_val if lable in [0,2]]

train_loader = torch.utils.data.DataLoader(cifar2,batch_size = 64,shuffle = True)
n_out = 2 #模型最后输出只有两个特征
model = nn.Sequential(
    nn.Linear(3072,512),nn.Tanh(),nn.Linear(512,n_out),nn.LogSoftmax(dim = 1)
)
loss_fn = nn.NLLLoss()
learning_rat = 1e-2
n_epochs = 100
import torch.optim as optim
optimizer = optim.SGD(model.parameters(),learning_rat)
for epoch in range(n_epochs):
    for imgs,labels in train_loader:
        batch_size = imgs.shape[0]
        out = model(img.view(batch_size,-1))
        loss = loss_fn(out,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d,Loss : %f" %(epoch,float(loss)))
#这是一个很简单的神经网络，只有两层。中间夹一个激活函数。
```

其中，imgs是64$\times$ 3$\times$ 32$\times$ 32的张量，labels 是大小为64的张量。



可以看到输出快了很多，而且结果也还不错。

- 全连接网络的局限性：

1. 全连接网络不是移动不变的。一个训练出来的模型可能能识别出在一个位置的飞机，但是如果另一张图片还是这个飞机，但是移动了位置可能就识别不出来了。

