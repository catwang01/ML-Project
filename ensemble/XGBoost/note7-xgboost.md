[toc]

# ML 学习笔记 7 xgboost

虽然都将 XGBoost 和 GDBT 放起来比较。但是在学习完两个模型之后，感觉两个的思路还是不太一样的。XGBoost 感觉更像 ID3、C4.5 之类的决策树。

只不过有了不同的增益度量，还有系统上的优化。

## 修改CART树 

XGBoost 的论文一开始就做了一系列的推导，目的是为了得出新的增益度量来代表原来的 CART 树中的 Gini Index 或样本方差。

因此，个人认为 XGBoost 的一个贡献是提出了新增益指标计算方法。这个增益指标和其它的增益指标如 Gini Index 不同的是，这个增益指标是在 Boosting 过程中推导出来的，可以说是对 Boosting 方法有着天然的亲合性。虽然也可以在 Boosting 的过程中使用原来的增益指标如 Gini Index，但是这些指标得到的 loss 会比 XGBoost 的新增益指标得到的 loss 要大，因此效果没有 XGBoost 好。

### CART 树

首先，一棵决策树可以看做是对样本空间的一个划分，其每个叶结点都对应着一个区域 $R_{j}$，因此，一棵有 T 个叶结点的CART树可以表示为

$$
f(x) = \sum_{j=1}^{T} w_{j}I(x \in R_{j})
$$ 

将上面的表示进行一些小小的变化。引入函数 $q(x_i)$  表示第 $i$ 个样本在所在的结点的下标。

因此，$f(x_i)$  可以表示为

$$
f(x_i) = w_{q(x_i)}
$$ 

因此，$f(x)$  可以看做有两个参数，一个是 $q(x)$ ，它表示树的结构。另一个是 $w_{j}$  表示每个结点上的权重。

### 新增益度量的推导

首先，XGBoost 使用的也是可加模型 + 前向分布算法。这个 GDBT 相同

其模型的形式为 

$$
\hat{y}(x)  = \phi(x) = \sum_{k=1}^{K} f_k(x)
$$ 

其中 $f_k(x)$ 是基函数，在 XGBoost 中选用 CART 树作为基函数。

首先，给出 XGBoost 的目标函数

$$
L(\phi) = \sum_{i=1}^{N} l(y_i, \hat{y_i}) + \sum_{k=1}^{K} \Omega(f_k)
$$ 

其中，$\Omega(f)$ 是正则项，其中包含两项，一项是对树的复杂度的惩罚，即对叶结点个数的惩罚，这里记 T 为叶结点的个数。另一个是对每个结点的输出分数的惩罚，这里记 $m$ 为每个叶结点上的输出分数。 

$$
\Omega(f) =  \gamma T + \frac{1}{2} \| w \|^2
$$ 

记第 $t$ 个分类器为

$$
\hat{y}^{(t)}(x) = \sum_{i=1}^{t} f_i(x)
$$ 

则对应的第 $k$ 步的损失为 

$$
\begin{aligned}
    L^{(t)} &= \sum_{i=1}^{N} l(y_i, \hat{y}^{(t)}_{i}) + \Omega(\hat{y_i}^{(t)}) \\
    &=  \sum_{i=1}^{N} l(y_i, \sum_{k=1}^{t} f_k(x_i)) + \Omega(\sum_{k=1}^{t} f_k) \\
    &=  \sum_{i=1}^{N} l(y_i, \hat{y}^{(t-1)} + f_k(x_i)) + \Omega(\sum_{k=1}^{t} f_k) \\
    &=  (1) + (2)
\end{aligned}
$$ 

其中

$$
\begin{aligned}
    (1) &= \sum_{i=1}^{N} l(y_i, \hat{y}^{(t-1)} + f_t(x_i)) \\
        &= \sum_{i=1}^{N} l(y_i, \hat{y}^{(t-1)} + f_t) \\
    (2) &= \Omega(\sum_{k=1}^{t} f_k) \\
\end{aligned}
$$ 

对(1)做Taylor Expansion展开到2阶

$$
\begin{aligned}
    (1) &= \sum_{i=1}^{N} l(y_i, \hat{y}^{(t-1)} + f_t) \\
    &\simeq \sum_{i=1}^{N} 
    [ l(y_{i}, \hat{y}^{(t-1)})
     + \frac{\partial l(y_i, y)}{\partial y}  | _{y=\hat{y}^{(t-1)}} f_t
     + \frac{1}{2} \frac{\partial^2 l(y_i, y)}{\partial y^2}  | _{y=\hat{y}^{(t-1)}} f_t^2 ] \\
    &= \sum_{i=1}^{N} 
    [ l(y_{i}, \hat{y}^{(t-1)})
     + g_i f_t
     + \frac{1}{2} h_{i} f_t ^2 ]\\
         &= constant + \sum_{i=1}^{N} (g_{i} f_{t} + \frac{1}{2} h_{i} f_t^2) \\
\end{aligned}
$$ 

考虑正则项

$$
\begin{aligned}
    (2) &= \Omega(\sum_{k=1}^{t} f_t) = \sum_{k=1}^{t} \Omega(f_k)  \\
    &= constant + \Omega(f_t) \\
\end{aligned}
$$ 

由于 $f_1, \ldots, f_{t-1}$  已经在之前的步骤中计算出来了，因此可以看做常数，故只有 $\Omega(f_t)$ 是需要优化的。

因此，去掉常数项后，XGBoost 的损失函数可以化简为

$$
\begin{aligned}
    \tilde{L}^{(t)} &= \sum_{i=1}^{N} (g_{i} f_{t} + \frac{1}{2} h_{i} f_t^2) + \Omega(f_t) \\
    &= \sum_{i=1}^{N} (g_{i} f_{t} + \frac{1}{2} h_{i} f_t^2 )+ \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 \\
\end{aligned}
$$ 

考虑到前面的求和是 $\sum_{i=1}^{N}$ 是按照样本编号对样本求和，而后面的求和是 $\sum_{j=1}^{T}$  是按照叶结点编号对样本求和。

下面我们想要将前一个求和也转换为对叶结点上的样本求和，这样做的好处是，$f_t$ 在叶结点 $j$ 上的输出就是  $w_{j}$ ，这样可以和之后的输出合并。

利用[之前](#cart-树)说的 $q(x)$ 上面的函数可以进一步化简

$$
\begin{aligned}
    \tilde{L}^{(t)} &= \sum_{i=1}^{N} (g_{i} f_{t} + \frac{1}{2} h_{i} f_t^2 ) + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 \\
    &= \sum_{j=1}^{T} \sum_{w(x_i)=j} (g_{i} f_{t} + \frac{1}{2} h_{i} f_t^2 ) + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 \\
    &= \sum_{j=1}^{T} \sum_{w(x_i)=j} (g_{i} w_{j} + \frac{1}{2} h_{i} w_{j}^2 ) + \gamma T + \frac{1}{2}\lambda  \sum_{j=1}^{T} w_j^2 \\
    &= \sum_{j=1}^{T} w_{j} \sum_{w(x_i)=j} g_{i}  + \frac{1}{2} w_{j}^2 \sum_{w(x_i)=j}h_{i}) + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 \\
    &=  \sum_{j=1}^{T} (w_{j} G_{j} + \frac{1}{2} w_{j}^2 H_{j}  ) + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 \\
    &=  \sum_{j=1}^{T} (w_{j} G_{j} + \frac{1}{2}(H_i + \lambda ) w_{j}^2 ) + \gamma T
\end{aligned}
$$ 

其中 $G_j = \sum_{w(x_i)=j} g_{i}$ 且 $H_j = \sum_{w(x_i)=j}h_{i}$ 表示在叶结点 $j$  上的一阶导数和二阶导数的和。

$\tilde L^{(t)}$ 本来有一个参数要优化，就是第 $t$ 步的基函数 $f_t$ 。而 $f_t$ 实际上又是取决于树的结构 $q(x)$  和权重 $w_j$  。

因此 $\tilde L^{(t)}$ 有两个参数需要优化：树的结构 $q(x)$ 和 每个叶结点上的权重 $w_{j}$  

下面分别优化这两个参数。

1. 固定 $q(x)$，对 $w_{j}$ 进行优化。注意到 $\tilde L^{(t)}$ 对于每个 $w_{j}$  实际上是一个二次函数，因此最值在 $-\frac{b}{2a}$  处取得，即

$$
w_{j}^{\star} = - \frac{G_{j}}{H_{j} + \lambda}
$$ 

2. 将 $w_{j}^{\star}$  代入， $\tilde L^{(t)}$变成 $q(x)$ 的函数，即
$$
\tilde L^{(t)}(q) = - \frac{1}{2} \sum_{i=1}^{T} \frac{G_{j}^2}{H_{j} + \lambda} + \gamma T
$$ 

上面的这个式子，可以用来作为一个指标来衡量 $q(x)$ 的好坏。这个式子越小，$q(x)$ 越好。

通常无法枚举所有的结构 $q$ ，因此使用贪心算法，从一个结点开始生成一棵二叉树。

假设一个结点上的样本的下标集合为 $I$ ，考虑将其分裂为两个结点  $I_L$ 和  $I_R$  。利用上面的损失函数，可以定义增益指标为 

$$
L_{split} = \frac{1}{2} \left[
  \frac{G_{L}^2}{H_{L} + \lambda} + 
  \frac{G_{R}^2}{H_{R} + \lambda} -
  \frac{G_I^2}{H_I + \lambda}
\right] - \gamma
$$ 

用这个指标来代表 CART 树中原来的 Gini index 或样本方差来生成 CART 树。

注意上式中 $G_I = G_L + G_R$ ,  $H_I = H_L + H_R$ ，利用这个关系可以简化计算。

### 防止过拟合：shrinkage 和列采样

为了进一步防止过拟合，XGBoost还使用了 shrinkage 和 column sampling 。

1. shrinkage 就是指在更新加法模型时，添加一个超参数 $\eta$ 

$$
y^{(t)}(x) = y^{(t-1)}(x) + \eta f_t(x)  
$$ 

2. 列采样

列采样是参考随机森林的构建，每次构建一棵树时，使用的特征是随机抽出来的。这种方法一方面可以减少计算量，另一方面可以减少过拟合。

## 树的分裂——寻找切分点

### 寻找分割点

#### Exact Greedy Algorithm

如何考虑在一个结点上是否应该往下分裂呢？一个比较简单的算法是对于这个结点上的所有样本都当作分割点进行分割。对于所有的特征都进行上面的计算，就可以得到最优的分割点以其对应的特征。

假设有 100 个样本，只有一个特征，如果用 Exact Greedy Algorithm，需要将每个样本都当作分割点计算一下增益，因此需要进行大约 100 次计算。

如果样本量很大的话，这种方法的时间复杂度会比较高。

这种方法被称为 Exact Greedy Algorithm，如下

```
Input: 结点上的所有样本
G = 结点上的所有样本的一阶梯度和
H = 结点上的所有样本的二阶梯度和
for feature_k in 该结点上的的所有特征
    for j in sorted(I, by feature_k)
        G_L = G_L + g_j, H_L = H_L + h_j
        G_R = G - G_R, H_R = H - H_L
        计算增益 score，如果大于当前增益，就更新最优分割点
     repeat
repeat
Ouput: 最优分割点
```

#### Approximate Greedy Algorithm

Exact Greedy Algorithm 尝试将所有的样本都当作分割点来进行测试。计算量当然大。一个很直观的想法就是增大样本的颗粒度，将相近的样本放在一起，当作一个整体。比如有 100 个样本，如果用 Exact Greedy Algorithm，需要将每个样本都当作分割点计算一下增益，因此需要进行大约 100 次计算。

而如果每10个样本放在一起，相当于有10个桶。则只需要进行 10 次计算。

这个就是 Approximate Greedy Algorithm

Approximate Greedy Algorithm 在实现起来有两种。一种是 local 和 一种是 global 的。

两者的区别是 local 的每次划分完需要再次划分。而 global 的只划分一次。

可以预见，local 的会更新精确，但是需要更多计算； global 没有 local 的精确，但是需要的计算也更少。

论文中给出了下面的结论来说明 Approximate Greedy Algorithm 的效果

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200425111726.png)

 其中 eps 可以理解为桶的大小。 eps 越小，桶的大小越小，分的桶的数量越多； eps 越大，桶的大小越大，分的桶的数量越少。

从上面可以看到下面的几个结论：
1. 在 eps 相同时， local 的效果比 global 好。
2. 通过调用合适的参数 global 也可以达到与 local 相同的效果。如图中 global 在 eps = 0.05 时的效果与 local eps = 0.3 的效果相当。
3. local 和 global 可以较好的逼近 Exact Greedy Algorithm


#### Weighted Quantle Sketch

一般来说，使用分位数就可以进行分桶了。但是，作者提出了加权分位数分桶的逻辑。


这个部分貌似挺难理解的。之后再看。先大概理解一下为什么要加权。


目标函数为

$$
\tilde L^{(t)} = \sum_{i=1}^{n}\left[g_{i} f_{t}\left(x_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(x_{i}\right)\right]+\Omega\left(f_{t}\right)
$$

配方后得到

$$
\tilde L^{(t)} =  \sum_{1}^{n}\left[\frac{1}{2} h_{i}\left(f_{t}\left(x_{i}\right) - (-g_{i} / h_{i}) \right)^{2}\right]+\Omega\left(f_{t}\right)+\text {constant}
$$

>注意，原paper中的这个部分的符号貌似写错了。虽然结论是正确的

可以看到，在每一项之前有一个权重。

# Todo - 2020-04-25 13:40 -- by ed

现在有一个问题，就是这个权重为什么会导致不平衡？导致什么东西不平衡？划分样本点的时候是根据什么来划分的？我原来以为是根据在某个特征上的取值排序后来划分的。但是这样看来，貌似又是根据二阶排序来进行划分的？

这个貌似还需要看其它文献。

可以之后看看 [2]

### 缺失值处理 Sparsity-aware Split Finding

文章中列举了可能导致缺失值的几种情况
1. 样本在某个变量上本身就确实数据
2. 虽然有值，但是都是0
3. one-hot encoding导致的

XGBoost 包含了缺失值处理的功能。其实很简单。就是分别尝试将这一堆缺失值划分为左子结点和右子结点，看那个效果好，就将缺失值划分到那个结点上去。

>个人感觉这种处理方法是很暴力的。因为它将这堆缺失值都分成到了同一个结点了。可是，有可能这堆缺失值实际上属于不同的叶结点呢？
但是不管怎么来说，有总比没有好。


## 系统优化

### Column block 结构

XGBoost 使用 Column block 结构来减少排序消耗以及实现并行。

因为在算法中，花费时间最多的是排序。

因此，论文中提出了使用 Block 结构来存储数据。Block结构有下面的几个特点：
1. 以CSC进行存储，不存储缺失值，
2. 在存储时对特征进行了排序，因此在之后使用时不需要排序（排序时不需要考虑有缺失值的样本,因为也没有办法排序）
3. 在特征中存储了对应的样本索引，可以使用这个索引来找到样本对应的梯度。
4. 一个 block 中可以存储多个特征值。

论文中的这张图可以清楚地解释 Block 的结构。

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200425141404.png)

可以看到，这种 block 结构可以分块并行，此时每个 block 对应一堆样本。
也支持列采样，只需要将 block 中对应的列提取出来即可。

### Cache-aware Access

block 的一个缺点是将数据的依赖取消了，会导致 cache miss。这部分内容比较底层。
大概就是说，在内存在读取第 $i$ 个样本的时候，实际上不光会读取第 $i$ 个样本，还会读取第  $i$ 个样本附近的数据放在缓存中，比如会将第 $i+1$  个样本也放在缓存中。下次再访问第 $i+1$ 个样本时发现在缓存中有，就直接从缓存中读取了，而不需要再从内存中读取。

而使用 block 结构后，我们对数据的访问变得不连续了。具体来说，block 中的数据是按照特征值大小排列的，我们会根据特征的值找到对应的 index ，然后读取其一阶导和二阶导。但是我们无法保证 block 中的相邻的数据对应的样本的 index  也相邻，因此可能导致 cache miss。

解决方法：
1. 对于 Exact Greedy Algorithm，使用缓存预取技术。
2. 对于 Approximate Greedy Algorithm，需要选择一个大小合适的 blocksize

### Out-of-core computation

当数据量比较大的时间，就需要考虑如何在内存外进行计算。

#### block compression

减少磁盘的吞吐量是一个很直接的想法，因此需要进行数据压缩。

1. 对于列来说，block 按列进行压缩，并在读取时使用独立的线程解压。
2. 对于行来说，使用了16bit 2个bytes 的int来存储index。并且只存储偏移量，这样可以减少内存消耗。（因此每个block最多存 $2^{16}$ 个样本。因为int是16位的）

#### block sharding

将数据划分到不同的磁盘上，可以采用并行来同时读取，以提高磁盘吞吐率

# References
1. [[1603.02754] XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) 
2. [XGBoost解读(2)--近似分割算法 • YXZF's Blog](https://yxzf.github.io/2017/04/xgboost-v2/)
3. [BackyardofAbela/EnsembleLearning: 包括决策树和随机森林进行离职人员预测，Xgboost和lightGBM的应用](https://github.com/BackyardofAbela/EnsembleLearning) 
