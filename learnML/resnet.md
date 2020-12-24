[toc]

# Paper 文献阅读一： Resnet

## 网络退化 degradation 问题

文章开头在说，网络的深度可以提高网络的预测效果。可是深层的网络不怎么好训练，不过这个问题已经很大程度上被 BN + SGD 解决了。

之后，作者提出了神经网络的退化（degradation）问题：深层次的网络在training data 和 testing data 上的表现都不如浅层网络：

![f710666c236c11c8762664753c071ccd.png](evernotecid://7E3AE0DC-DC71-4DDC-9CC8-0C832D6C11C2/appyinxiangcom/22483756/ENResource/p12223)

很显然，这个问题不是由于过拟合导致的，因为深层网络即使在训练集上都搞不过浅层网络！

## 退化问题体现的问题

作者是这样解释退化问题的：网络的退化说明了网络结构（system不知道是不是这个意思）的估计难度是不一样的。

>The degradaiton (of training accuracy) indicates that not all systems are similarly easy to optimize.

之后，作者进行了进一步的实验来验证了这个结论。他们训练了一个浅层网络。同时基于这个浅层网络构造了深层网络，构造的方法很简单：就只是在训练好的浅层网络的基础上添加一堆恒等映射，用这个网络再去训练，得到的网络效果至少也应该和原来的浅层网络一样。（疑问：这个identity mapping是怎么加的没有想清楚。是在直接在浅层网络的后面或者中间加一堆隐藏层，权重W取为1吗？）

>Let us consider a shallower archtecture and its deeper counterpart that adds more layers onot it.There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model. The existence of this constructed solution indictes that a deeper model should produce no hgher training error than its shallower counterpart.

degradation 现象反映出不同的网络结构的训练难度是不一样的。

设对于一个真实的映射 $H(x)$ 的估计为 $\hat{H}(x)$，我们希望通过网络来学习出的映射 $F(x)$ 来估计 $H(x)$，即 
$$F(x) = \hat{H}(x)$$

**假设**估计残差 $H(x)-x$ 的难度要比估计 $H(x)$ 简单，那我们直接去估计 $H(x)-x$ 就好了。即此时 $F(x) = \hat{H}(x) - x$，由此得到

$$\hat{H}(x) = F(x) + x$$

因此我们相当于间接得到了 $H(x)$ 的估计 $\hat{H}(x)$


## 为什么残差更好估计？

上面推断有一个假设： **估计残差 $H(x)-x$ 的要比估计 $H(x)$ 简单**。

可是这个假设是不是正确的？

### 极端情况 H(x) = x

首先，作者认为，degradation现象表明，对于非线性的网络来说，identity mapping 是比较难估计。

>The degradation problem suggests that the solvers might have difficulties in approximaitng identity mappings by multiple nonlinear layers.

也就是说，极端一点，如果真实的 $H(x)$ 是identity mapping，即 $H(x) = x$，则 $H(x)$ 是比较难估计的。（因为上面刚说过 degradation 现象表明 identity mapping 是比较难估计的）

而其残差 $H(x) - x = 0$ 是很好估计的，这只需要通过训练将所有的权重都变成0就可以了。

也就是说，在 $H(x) = x$ 这种极端情况下，残差要比原映射更好估计。

>With the residual learning reformulation, if identity mappings are optimal, the solvers may simply drive the weights to the multiple nonlinear layers toward zero to approach identity mappings.


### 其它情况

上面说明了极端情况下，残差的确更好估计。而在一般情况，作者认为直接估计残差也是有好处的。因为这样相当于给出了一个参考值，网络会从 identity mapping 开始去寻找最优解，而不是从 zero mapping 开始去寻找最优解，实验表明这会有好处。


>In real cases, it is unlikeyly that identity mappings are optimal, but our reformulation may help to precondition the problem. If the optimal function is closer to an identity mapping than to a zero mapping, it should be easier for the solver to find the perturbations with reference to an identity mapping, than to learn th function as a new one. We show by experiments that the learned residual functions in general have small responses, suggesting that identity mappings provide reasonablre preconditioning.

## 残差网络的 building block

### shortcuts

上面说了一通，到底要如何估计残差呢？作者提出了用 shortcuts 结构来实现残差网络：

![block](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200429202153.png)

用公式表示为

$$y = F(x, \{W_i\}) + x$$

其中 $F(x, \{W_i\})$ 是残差函数。

需要注意的是，先计算 $F(x, \{W_i\}) + x$后，再过激活函数。

### F(x)与x形状不同时怎么办？

上面的情况是 $F(x)$ 与 $x$ 的形状相同的情况，可以直接进行 elementwise addition。如果 $F(x)$ 和 $x$ 的形状不同时，可以在 x 上乘一个权重 $W_s$ 来调整 x 的形状，使其和 $F(x)$ 相同。此时公式修改为

$$y = F(x, \{W_i\}) + W_sx$$

注意到，即使 $x$ 与 $F(x)$  形状不同时，也可以对 $x$ 乘以一个方阵 $W_s$ 后再相加。作者在之后的实验中说明了这样做的性价比不高，虽然会有蚊子腿的性能提升， 但是增加了计算开销。**因此 $W_s$ 只有在二者形状不同时用来调整形状。**

>But we will show by experiments that tht identity mapping is sufficient for addressing the degradation problem adn is economical， and thus $W_s$ is only used when matching dimensions.

**不增加计算开销这一点还是很重要的**，因为作者认为残差学习的一大特点就是几乎不增加时间复杂度

>The shortcut connections introduce neither extra parameter nor computation complexity. 

### 可以用于巻积层吗？

由于巻积实际上就是一种特殊的乘法，因此也可以对巻积层进行残差学习。这时要注意，调整权重的操作需要变成一个巻积操作。

>We alse note that although the above notations are about fully-connected layers for simplicity, they are applicable to convolutional layers.  The element-wise addition is performed on two feature maps, channel by channel.

## 实验

### 优化策略

学习一下大牛的优化策略，或许能受到启发：

1. 每个卷积层后使用BN，不使用dropout
2. 优化器为SGD 初始lr为0.1，在error plateaus处乘以0.1，使用 weight decay，decay rate 为 0.0001，momentum = 0.9
3. batch_size = 256, 迭代了 $60 \times 10^4$ 次

其它的实验结果没有什么好说的，大概就是残差网络牛B呗！这里上一张图

![net structure](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200429202324.png)

### Deeper Bottlenectk Architectures

从上面的图中可以看出，从 resnet50开始，building block 开变成了三层，分别是 1x1, 3x3, 1x1，的结构，这个实际上是bottleneck结构，主要是用来在不增加复杂度的前提下增加深度的。

>Deeper non-bottleneck Resnets also gain accuracy from increased depth, but are not s economical as the bottleneck ResNets. So the usage of bottleneck designs is mainly due to practical considerations.

从上面的图也可以看出，ResNet34 没有使用 bottleneck 结构，计算量为 $3.6 \times 10^9$，而 ResNet50使用了 bottleneck 结构，其层数只增加了一半，而计算量只为 $3.8 \times 10^9$，和 ResNet34相差无几。

下面是bottleneck的结构

![bottleneck](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200429202243.png)


