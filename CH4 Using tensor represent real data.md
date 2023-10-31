神经网络中的所有操作就是张量之间的操作。神经网络之间的所有参数都是张量。比如权重和偏置。本章主要介绍如何获取一段数据，一段视频，或一行文本，并转化为张量 形式。

## 处理图像：

卷积神经网络的引入改变了计算机视觉，图像被表示为具有高度和宽度的规则网格中的标量集合。每个网格点叫像素，其可能有一个或多个标量。代表单个像素值的标量通常使用8位整数编码。

- RGB：

这是将颜色编码为数字的一种方法。 颜色由三个数字定义。分别代表红、绿、蓝。每一个颜色通道可以看成一个灰度强度图。

---

- 加载图像文件：

Python提供许多处理图像的方法，比如可以使用imageio模块来加载PNG格式的图像。处理的结果具有三个维度。两个空间尺寸维度，第三个维度对应于RGB通道，得到的是数组，可以使用torch.from_numpy转为张量。在P yRorch模块中要求张量排列为CHW(通道，高度，宽度)。为了满足这个要求，可以使用permute交换张量各个维度的顺序。

```python
import torch
import imageio

img_arr = imageio.imread('/Users/apple/Desktop/python/picture/pictry1.png')
#导入了图片，此时是数组，而且维度排序是高度，宽度，通道。不符合要求。
img1 = torch.from_numpy(img_arr)
img = img1.permute(2,0,1) #没有复制数据，img1和img底层的数据一样的。此操作开销很低。改变img1,img中一个会改变另一个。
print(img.shape)
```



输出为torch.Size([3, 1321, 1200])， 满足要求



为了批量处理多幅图片，需要沿着第一个维度批量存储图像。获得NCHW的张量，N表示图片的编号。我们可以预先分配一个适当大小的张量，并使用从目录中加载的图像填充它。

``` python
import torch
import imageio

batch = torch.zeros(3,3,256,256,dtype = torch.uint8)#批量导入三张图片
# 此处只需要使用torch.uint8这一数据类型。用3个0至255来区分一个像素点的颜色是足够的。

import os
datas_path = '/Users/apple/Desktop/python/picture/catstry'
filenames = [name for name in os.listdir(datas_path) if os.path.splitext(name)[-1] == '.png'  ]
#os.path.splitext(name) 把name的文件名分为两部分，文件名和扩展名，返回一个tuple.这里确定了name的扩展名是png。
i = 0
for name in filenames:
    img_arr = imageio.imread(os.path.join(datas_path,name) )
    img1 = torch.from_numpy(img_arr)
    img = img1.permute(2,0,1)
    img = img[:3]#这里我们只保留前3个通道，有时图像还有一个表示透明度的 alpha 通道，但我们的网络只需要 RGB 输人
    batch[i] = img
    i=i+1
```

-----

- 正规化数据：

神经网络通常使用浮点数张量作为输入，且当输入的范围在0~1或者-1~1时，神经网络表现的训练性能最好。这是由Pytorch模块构建的方式决定的。

因此我们经常把张量转换为浮点数类型并且对像素的值归一化。可以先使用float方法把张量变为浮点数型，然后在做归一化操作。

在本节的例子中，可以直接对数据除以255来进行正规化。另一种方法是使用torch.mean，torch.std计算数据均值和标准差，然后归一化。



## 三维图像：体数据

以CT为例。CT扫描中，除了三维的空间数据，还需要一个强度数据，代表身体不同部位脂肪，水，肌肉，骨骼的密度。用5维张量表示，NCDHW。在上一节的例子的基础上增加了张量D。

可以使用imageio中的volread()函数加载一个CT扫描版本，转换为array数据。然后处理和上一节一样，转化为张量，交换顺序，归一化。

---



## 表示表格数据：

电子表格，CSV文件，数据库等都是一张表，每行包含一个样本或记录，而每一列则是样本的一部分信息。

对于电子表格，需要假设样本在表格中的出现顺序没有意义。表格的列往往包含不同的数据类型。比如一个学生的名字和考试成绩分别是字符串和浮点数。但是PyTorch的张量的数据类型全是同一种。

PyTorch加载CSV文件有三种方法：

​	-- Python自带的CSV模块

​	--NumPy

​	--pandas

主要介绍第二种：

```python
import numpy as np
import torch
torch.set_printoptions(edgeitems=2, precision=2, linewidth=75)
import csv
wine_path = "/Users/apple/Desktop/python/csv/winequality-white.csv"
# csv存放的路径
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";",
                         skiprows=1)
#这里我们只规定了二维数组的类型（32 位浮点数）、用于分隔每行数据的分隔符以及不读取第1行（因为它包含列名)。
wineq = torch.from_numpy(wineq_numpy)
print(wineq.shape)
```



数据的连续值，序数值，分类值

1. 连续值是有直观的现实意义的。比如物体的重量，路的长度等。连续值可以进一步划分。例如，可以一说一个物体的质量是另一个的两倍，这样说是有意义的。所以质量这一连续值被称为比例尺度。像说6:00是3:00的两倍是没有意义的。所以时间这样的连续值只提供了一个区间尺度，没有比例尺度。
2. 序数值表达对数据的严格排序，而序数值具体的值是多少不重要。比如，把小杯，中杯或者大杯分别映射为1，2，3或者2，4，6都是可以的。我们只能对这些数据进行排序，不能进行数学运算。不能因为（1+3）/2 = 2就认为大杯和小杯平均是中杯，这是不合理的。
3. 分类值，即没有排序意义，也没有数学意义，只用于分类。由于分类值没有实际意义，也叫做名义尺度。

----

- 葡萄酒的分数的表示：

可以将分数视为连续变量，把它当为一个实数，也可以把它当为一个标签。



## 处理时间序列：

教材以华盛顿区的自行车共享系统每小时的自行车租赁数量为例，第一个轴以日期为索引，第二个轴以表示一天中的小时，独立于日期。第三个轴表示不同的数据列，包括天气温度等。

```python
import numpy as np
import torch
import csv 
bikes_numpy = np.loadtxt( "/Users/apple/Desktop/python/bike-sharing-dataset/hour-fixed.csv", dtype=np.float32, delimiter = ",", skiprows=1,converters={1: lambda x: float(x[8:10])} )
bikes = torch.from_numpy(bikes_numpy)
```

在这样的时间序列数据集中，行表示连续的时间点。我们想要把2年的数据集分成更细的观察周期，如按天划分。这样就有了序列长度L，样本数量N的集合C。这个例子中的时间序列是一个3维张量，NCL。 C是17，是原先表的列数，L是24，是每天的小时数，N是天数。

（这个例子中的数据集已经按照时间排序，如果没有排序，可以使用torch.sort()来排序）



目前bikes的Size还是（17520，17），还不是我们要的三维数组。使用view函数来调整步长，从而调整张量的形状。

```python
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
# view函数可以通过改变张量步长的情况下，在不改变张量底层数据是，改变张量的形状。-1表示这个维度的数据被其它数据推导出来。bikes.shape[1] = 17
daily_bikes = daily_bikes.transpose(1, 2)
#之前得到的是NLC，我们想要NCL，于是对做一个转秩
print(daily_bikes.shape, daily_bikes.stride())
```

输出的结果是torch.Size([730, 17, 24]) (408, 1, 17)。

- 准备训练；

我们先要研究天气状况变量。这个变量是有序的。它有四个级别。1表示晴天，4表示大雨/大雪。我们暂时只关注第一天的数据。

```python
first_day = bikes[:24].long()#截取第一天的数据
weather_onehot = torch.zeros(first_day.shape[0], 4)
#first_day.shape = torch.Size([24, 17])。4是因为有四种天气状况。该维度每一个值为1表示是这种天气，0表示不是。这就是独热编码的形式。
weather_onehot.scatter_(dim=1, index=first_day[:,9].unsqueeze(1).long() - 1,value=1.0)
#unsqueeze作用是在指定位置加上维数为1的维度。
#把1，0映射到weather_onehot中，index决定weather_onehot上第1维，也就是列处的位置。第i行的第index[i]个位置变为1.0。-1是因为天气从1至4，而weather_onehot的列坐标从0至3.
print(first_day[:,9],first_day.shape)
print(weather_onehot)
```



然后，可以使用cat（）函数将weather_onehot与原始张量bikes接在一起。这里按维数1拼接，即把weather_onehot接为bikes后四列。这里需要保证，也保证了其它维数大小形同，即weather_onehot和bikes的行数相同。

上述只是以一天举例。可以对重塑的张量daily_bikes也做同样的事。

```python
import numpy as np
import torch
import csv 
bikes_numpy = np.loadtxt( "/Users/apple/Desktop/python/bike-sharing-dataset/hour-fixed.csv", dtype=np.float32, delimiter = ",", skiprows=1,converters={1: lambda x: float(x[8:10])} )
bikes = torch.from_numpy(bikes_numpy)
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
# view函数可以通过改变张量步长的情况下，在不改变张量底层数据是，改变张量的形状。-1表示这个维度的数据被其它数据推导出来。bikes.shape[1] = 17
daily_bikes = daily_bikes.transpose(1, 2)
#之前得到的是NLC，我们想要NCL，于是对做一个转秩。现在形状是NCL。
daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4,daily_bikes.shape[2])#建立24行4列的全零矩阵。
daily_weather_onehot.scatter_(1,daily_bikes[:,9,:].long().unsqueeze(1) - 1,1.0)
daily_bikes = torch.cat((daily_bikes,daily_weather_onehot),1)
print(daily_bikes.shape)
#通道增加了4，用于表示四种天气
```



独热编码是处理天气数据的一种方法，另一种是将天气看为连续变量的特殊值。我们只需转变其变量。（不要改变序关系，天气是有序数）

---



## 表示文本：

深度学习在自然语言处理（Natural Language Processing，NLP）领域取得了巨大的成功。特别是使用那些反复消费新输入和以前模型输出的组合的模型。这样的模型叫做循环神经网络。（Recurrent Neural Network，RNN）本节重点是如何把文本转换为数字张量。

- 将文本转化为数字：

`with open('/Users/apple/Desktop/python/bookdata-ch4/jane-austen/1342-0.txt',encoding = 'utf8') as f: text = f.read`

方法是使用独热编码字符转换为数字：

我们为每个字符提供独热编码。每个字符由一个长度等于编码中不同字符数的向量表示。该向量除了与编码中字符位置对应的索引为1，其它都为0.

以其中一行为例如何转换。

```python
import torch
import numpy as np
with open('/Users/apple/Desktop/python/bookdata-ch4/jane-austen/1342-0.txt',encoding = 'utf8') as f: text = f.read()
lines = text.split('/n')
line = lines[100]
letter_t = torch.zero(len(line),128)#这是英文文本，只需要到128就可以表示所有字符
for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0  
    letter_t[i][letter_index] = 1  
```

---

- 也可以使用独热编码对一整个单词编码，由于单词种类很多，这不一定实用。先对句子做一些特殊处理，比如去掉连接符号等：

```python
import torch
import numpy as np
with open('/Users/apple/Desktop/python/bookdata-ch4/jane-austen/1342-0.txt',encoding = 'utf8') as f: text = f.read()
lines = text.split('\n')
line = lines[100]
def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace('\n',' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list

words_in_line = clean_words(line)
print(words_in_line)
```

接下来创建一个单词到索引的映射：

```python
word_list = sorted(set(clean_words(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}
print(len(word2index_dict), word2index_dict['impossible'])
```

输出结果是7261 和 3394。说明这篇文章中共有7261个不同的单词，其中"impossible"是第3394个。对之前那一行，有

```python
word_t = torch.zeros(len(words_in_line), len(word2index_dict))
for i, word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_t[i][word_index] = 1
    print('{:2} {:4} {}'.format(i, word_index, word))
    
print(word_t.shape)
```

从而完成对这一行句子的数据转换



对单词编码数据量比较大，但是单词表达的信息更多。多数情况下，折中的办法基于单词分割。比如字节对从单个字母的字典开始，迭代地将观察到的最平凡的字节对添加到字典中，直到它到达指定的字典大小。

---

- 文本嵌入：

使用独热编码的问题很明显：当要编码的数据量很大时，往往不好用。为了将编码压缩到一个更易于管理的大小，并限制其增长，可以用浮点数向量来代替很多0和1的向量。这种方法叫做嵌入。比如可以把单词表示为100个浮点数组成的向量。

例如，如果用二维的浮点数表示，可以这样表示：两个坐标分别映射到名词和形容词，例如名词是水果(0.0~0.33)、花(0.33～0.66)和狗(0.66~1.0)，以及形容词红色(0.0~0.2）、橙色(0.2～0.4)、黄色(0.4~ 0.6)、白色(0.6~0.6)和棕色(0.8～1.0)等。通常来说两个维度太少了。嵌入使得相似的单词聚在一起。而且对单词进行加减操作可以得到别的单词，例如苹果-红-甜+黄+酸，可以得到柠檬对应的向量。嵌入的算法比较复杂，通常先用独热编码，使用通常比较浅的神经网络生成嵌入，并将其用于下游任务。现代的嵌入模型BERT和GPT-2很精细，对上下文敏感。
