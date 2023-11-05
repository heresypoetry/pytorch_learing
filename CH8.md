# CH8 使用卷积：

## 卷积的作用：

卷积可以提供局部性和平移不变性，可以解决上一节中出现的问题。通过计算像素和其相邻像素的加权和。而不是与其它相距远的像素的加权和，来忽略掉那些彼此相距很远的元素。（本节的卷积指的都是离散卷积）



## 卷积的使用：

首先需要一个内核，在深度学习里面，内核一般是形状比较小的。比如我们可以有一个$3\times3$矩阵作为内核，然后对一个一维的通道，$M\times N$的图像，把每个点（目前暂时不考虑周围一圈的点）更新为自己以及周围九个数据点的加权和。

就像nn.Linear一样，权重是不知道的，一开始随机给的。通过不断更新得到。

可以把卷积理解为权重在除个别像素点外的地方都为0 。所以可以大幅度减少模型的张量。

torch.nn分别提供一维，二维，三维的卷积，分别叫nn.Conv1d ,nn.Conv2d ,nn.Conv3d, 分别可用于时间序列，图像，体数据和视频。本节我们使用的是二维的卷积。

-  填充边界：

由于对边界的像素的没有进行和周围像素的加权求和，使得我们的输出图像要比输入图像小一圈。即决办法是在边界周围创造一些重影像素（ghost pixel）。就卷积而言，这些重影像素的值为0 。这样即使在边界处也能计算出卷剧的输出。最后输出的图像和输入的图像大小完全相同。

-  用卷积检测特征：

如果用

```Python
with torch.no_grad ():
conv.bias.zero(）
with torch.no_grad():
conv.weight.fi11_(1.0 / 9.0)
```

将卷积的权重全调为1/9 , 这样图像虽然变模糊了，但是比较光滑。又比如，可以将卷积核写为：

```
[[-1.0,0.0,1.0]
[-1.0,0.0,1.0]
[-1.0,0.0,1.0]]
```

这样的核是一个边缘显示核，这样的核突出显示2个水平相邻区域的垂直边缘。

- 下采样：

这是将图像缩小的办法。比如，将图像缩小一半，就是取4个相邻像素作为输入，产生一个像素作为输出。比如有以下几种池化的方法：

1.取4个像素的啤平均值，这种方法现在已经不受欢迎了

2. 取4个像素的最大值，这叫做最大池化，是目前最常用的方法之一。但仍有丢弃四分之三数据的风险
3. 使用带步长的卷积。该方法只将每第N个像素纳入计算。这种方法很有前景，但是还没有取代最大池化法。

nn 模块自带最大池化方法，由nn.MaxPool2d模块提供。比如，如果想让图像缩小一半，则让MaxPool2d的参数（核大小）设置为2 。 

- 卷积和下采样结合：

![image-1](/Users/apple/Library/Application Support/typora-user-images/ml_learn_CH8_1.png)

- 输出像素的感受野：

当第2个3x3 的卷积核在其卷积输出中产生21 时，如图 8.8 所示，这基于第 1 个最大池化输出的左上角3x3 个像素。它们依次对应第1 个卷积输出中左上角的6×6个像素，而6×6个像素又由左上角 7×7个像素的第1次卷积计算而来。因此，第2个卷积输出中的像素会受到一个7×7的正方形的影响。第1个卷积还隐式地使用“填充〞的列和行来产生角落中的输出，否则，在第2 个卷积的输出中，我们将有一个 8×8的正方形的输入像素通知一个给定的像素（远离边界)。用花哨的语言来说，我们说一个给定的 3×3-conv、 2×2-max-pool、 3×3-conv 结构的输出神经元有一个8×8的感受野。

## 开始构建神经网络：

将上诉过程连接起来，可能会有下面的问题：

```Python
#指定padding = 1可以在i00上面和左边有一组额外的领域，这样即使在原始图像的角落也能计算出卷积的输出。当kernel_size= 3时，定义卷积核大小为3*3. 也可以直接写（3，3）。
#3,1分别表示输入的通道数和输出的通道数。
model = nn.Sequential(
    nn.Conv2d(3,16,kernel_size= 3, padding= 1),
    nn.Tanh(),
    nn.MaxPool2d(2),
    nn.Conv2d(16,8,kernel_size=3,padding=1),
    nn.Tanh(),
    nn.MaxPool2d(2),
    #下面开始全连接：
    #这里回出错，因为我们需要使用view()将输出的8*8*8张量转化为有512个元素的一维张量，但是可惜在使用Sequential时，没有任何显示的方法展示每个模块的输出。
    nn.Linear(512,32),
    nn.Tanh(),
    nn.Linear(32,2)
    )
#模型的结果得到了有8个通道，8*8个像素的图片，然后使用全连接得到概率。
```

## 子类化nn.Module：

我们可以把网络编写为一个子模块, 用这种方法来替代Sequential的方法，从而直接操作模块的输出。

```Python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 8 * 8 * 8) # <1>
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out
```

这样就根据我们的需要构造了Module类的一个子类，名叫Net类。Net类相当于以前用的nn.Sequential,但是它是显示地编写forward()方法。

- 组装完整的训练循环：

```python
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
from torch import optim
import torch.nn as nn
import datetime
data_path = '/Users/apple/Desktop/python'
cifar10 = datasets.CIFAR10(root= data_path,train = True, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))]))
cifar10_val = datasets.CIFAR10(root= data_path,train = False, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))]))
label_map = {0:0,2:1}
class_names = ['airplane','nird']
cifar2 = [(img,label_map[lable]) for img ,lable in cifar10 if lable in [0,2]]
cifar2 = [(img,label_map[lable]) for img ,lable in cifar10_val if lable in [0,2]]
train_loader = torch.utils.data.DataLoader(cifar2,batch_size = 64,shuffle = True)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 8 * 8 * 8) 
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out
model = Net()
optimizer = optim.SGD(model.parameters(),1e-2)
loss_fn = nn.CrossEntropyLoss()#交叉墒损失
def training_loop(n_epochs,optimizer,model,loss_fn,train_loader):
    for epoch in range(1,n_epochs+1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            outputs = model(imgs)
            loss = loss_fn(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train+=loss.item()
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))
training_loop(
```

- 输出测量精度：

为了直观地解释模型的准确度：在最后加入判断准确度的方法。

最后训练的模型有90%的准确率。

- 保存模型

可以使用torch.save来保存我们的模型到一个文件中之后使用, 使用时可以用loaded_model函数调出。

`torch.save(model.state_dict(), data_path + 'birds_vs_airplanes.pt')`
