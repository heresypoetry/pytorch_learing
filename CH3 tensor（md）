# Chpater 3 张量（Tensor）

张量是Pytorch 里面基本的数据类型。其与数学或者物理学中定义的张量明显不同。在Pytorch中可以将张量等价视为多维数组。张量是PyTorch包中的类。使用之前必须`import tensor`

除了PyTorch外，NumPy也是可以处理多维数组的Python中的包。PyTorch和NumPy具有很好的相互操作的特性，二者很多地方相似。PyTorch自己的特点包括可以在GPU上操作的很快。

- 张量的索引方式和多维数组的索引方式一样。例如`a = tensor([[1.0,2.0],[3.0,4.0]]).`则有`a[0][1] = 2.0`。

---



- 张量的创建方式：

```python
import torch
a = torch.ones(3)#创建3乘1的值全是1.0（32位浮点数）张量 
b = torch.ones(1,2,3)#创建1乘2乘3的全1.0张量
c = torch.zeros(1,2)#创建1乘2乘3的全0.0张量
```

---



- Python中的列表和PyTorch中的张量或者NumPy中的数组不同。列表中的各个元素在存储器中的位置是零散的，不连续的。而张量或数组存储的数据是连续存储的。

----



- 索引张量：

张量的索引方式和列表索引方式一样。比如对于列表a, a[:], a[1:], a[1:3], a[1:6:2] 分别表示列表a所有元素，从下标1至最后所有元素，从下标1至3.从下标1至6，步长为2。

对于张量points, points[1:]. points[1:,0], points[None]. 分别表示第一行之后所有行和所有列，第一行之后所有行的第一列，增加大小为1的维度。

可以运行以下实例:

```python
import torch
a = torch.zeros(3,3)
a[0][1] = 1.0;a[0][2] = 2.0
a[1][0] = 3.0;a[1][1] = 4.0
a[1][2] = 5.0;a[2][0] = 6.0
a[2][1] = 7.0;a[2][2] = 8.0
b = a;c = a;d = a;e=a;f=a
b = b[1:]
c = c[1:,:]
d = d[1:,0]
f = f[1:,1:]
e = e[None]
print(a);print(b);print(c);print(d);print(e)
```

----



- 张量上的操作：

可以参考官方给出的关于张量的方法和函{https://pytorch.org/docs/stable/torch.html}



此外，这里介绍几种常用的。

---



- 张量的命名：

```Python
import torch
a = torch.zeros(3,10,10)
a_named = torch.tensor(a,names = ['color','row','column'])
#对每个维度命名。例如张量a可以用来表示一张像素为10乘10的图片。#
#维度color有三个值是因为由RGB表示颜色#
#默认没有设定的名字为None。名字可以修改，可以把想取消的名字改成None#
b = torch.zeros(3,10,10,names = ('color','row','column'))
print(a.names);print(a_named.names)#查看名字
#当两个名称作为字符串相等，或者二者中有一个是None，则称两个名称匹配。#
#在实践中，可以尽量使用命名张量，避免不小心写出color+row这样多半不合理的操作#
print(a_named+b);print(a+b)#此时由于名称匹配，运算不会报错
rename_a_named = a_named.rename(row = 'rows',column = 'columns')
#a_named 的名称可以如上使用rename修改

#print(rename_a_named+b)
#如果运行这行代码，由于两个张量名字不同，不能一起运算，会报错。
```

----



- 张量数据类型：

张量构造函数通过dtype参数指定包含在张量中的数字的类型。默认的类型是32位浮点数。dtype可能的取值有torch.float64(=torch.double), torch.int32(torch.int), torch.bool等.

-----



- 张量的存储视图：

张量在计算机底层，存储位置是连续的。也就是说张量存储的方式实际上是一串连续的一维的数。可以使用storge()函数来访问张量的实际存储区。如：

```python
import torch
a = torch.zeros(2,3)
a[0][1] = 1.0;a[0][2] = 2.0
a[1][0] = 3.0;a[1][1] = 4.0
a[1][2] = 5.0;print(a.storage())
```

虽然这个张量自己是2乘3的，但是其在底层的存储器上，它是大小为6的连续数组，如果手动修改存储区的值，张量也会变化。为了表示一个张量，除了它的数据，也就是存储器中存储的这个连续数组以外，还需要知道张量的大小（或者说形状）。张量的大小表示张量在每个维度上各有多少个元素。

存储偏移量（storage offset）是指存储区对应于张量中的第一个元素的索引。步长（stride）指的是存储区中为了获得下一个元素需要跳过的元素数量。

例如a是tensor([[0., 1., 2.], [3., 4., 5.]] )。 存储区中表示为0. 1. 2. 3. 4. 5. 

[0., 1., 2.] 到[3., 4., 5.]，也就是0到3，到达下一行需要走3步。到达下一列，比如0.到1.，需要一步。因此。步长为[3,1]。

------



- 连续张量：

一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量很方便， 因为我们可以有效地按顺序访问它们。有一些操作也只能对连续张量起作用。可以使用is_contiguou（）方法判断是不是连续张量。可以使用contiguou（）把张量转换为连续张量。（转换的过程中直接改变了底层存储区元素的顺序）

```python
import torch
a = torch.zeros(2,3)
a[0][1] = 1.0;a[0][2] = 2.0
a[1][0] = 3.0;a[1][1] = 4.0
a[1][2] = 5.0;
#print(a.storage())#查看storage
a = a.t()#对a转置
#print(a.storage())#再次查看，发现转置后没有改变
#print(a.is_contiguous()) 返回False
a = a.contiguous()#转换为连续张量
print(a.storage())#查看storage，发现改变
```

-----



- 张量的设备属性。张量还有设备(device)概念，是张量的一个属性，表示张量数据在计算机上的位置。可以用to()方法把CPU创建的张量复制在GPU上。在此基础上，对张量的操作将在GPU上进行。需要注意的是，除非输出或访问得到的张量，计算出的结果不会返回CPU，而是留在GPU。为了将结果返回CPU，需要用to()方法。

-----



- 可以将Numpy的array转换为tensor。比如， 对于array a, 可以定义张量b, *b = torch.from_numpy（a）*。

  -----

  

- 序列化张量：

我们希望可以把张量保存在文件中。使用torch.save函数存储张量，使用torch.load()函数读取张量。torch.save函数第一个参数是张量，第二个参数是描述地址的字符串。torch.load()的参数是地址。

-----

比如， 对于array a, 可以定义张量b, *b = torch.from_numpy（a）*。
