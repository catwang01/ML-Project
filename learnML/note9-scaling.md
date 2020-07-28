[toc]

# 学习笔记 9 -- 归一化

## 什么是归一化？

这个词用的比较乱。有的人用归一化来指 min-max scaling 而标准化来指 standarize。
有的人用将这类似的方法统称为归一化。这里还是分开来说，就用 min-max scaling 和 standarize。然后用 scaling 来统称这两种变换。

### min-max scaling

$$
x_{norm} =  \frac{x - x_{min} }{x_{max} - x_{min}}
$$ 

1. 最大值与最小值非常容易受异常点影响，所以这种方法鲁棒性较差，只适合传统精确小数据场景。
2. min-max可以将数据放缩到一个特定的 range 上，这对图像处理问题来很有用，如将 0-255 放缩到 0-1

### standarize

$$
z = \frac{x - \mu}{\sigma}
$$ 

1. 在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景。
2. 和 min-max scaling 相比，不会将数据放缩到一个特定的 range 上。但是可以使 mean 为 0 ； variance 为 1.
3. 理论上输出范围为：$(-\infty, +\infty)$ 

## 归一化的作用

1. 不同变量往往量纲不同，归一化可以消除量纲对最终结果的影响，使不同变量具有可比性。比如两个人体重差10KG，身高差0.02M，在衡量两个人的差别时体重的差距会把身高的差距完全掩盖，归一化之后就不会有这样的问题。
2. 在涉及到计算点与点之间的距离时，使用归一化或scaling都会对最后的结果有所提升，甚至会有质的区别。
3. 加速梯度下降的收敛，如下面的图

![pic from 葫芦书](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200429195209.png)

上面的图在没有 standarize 的时候，梯度的方向没有指向最小值的方向，因此会比较震荡。而 stadarize 之后就指向了最小值的方向，因此更新会比较快。

4. 避免出现太大的数。导致数值问题。

## 需要scaling的模型

主要参考 [3]

### PCA

### 基于距离的算法 KNN、SVM、K-means

基于距离的算法，尤其是欧式距离，如果有一个特征的量纲特别大时，会在距离中占比很大。这样会导致不同的特征对距离的贡献值不同。
SVM 中计算了距离超平面的距离，因此也需要进行 scaling

### 添加正则项的模型

对于添加正则项的模型来说，变量的范围将会影响到他们对应系数受到什么程度的惩罚。因为方差大的变量对应的系数很小，因此它们会受到较小的惩罚。

如线性回归添加 L1 正则得到的 Lasso 和添加 L2 正则得到的 Ridge.

## 不需要scaling的模型

### 各种树模型

树模型的训练是主要根据各种信息增益来进行的节点分割。而节点的分割是基于对特征进行排序之后进行的。而归一化并不会影响特征的排序，因此是否归一化并不化影响到树模型的训练。

### 不带惩罚项的线性回归(逻辑回归)需要进行scaling吗？

[3] 认为逻辑回归不需要进行scaling。而这里提供了不一样的观点。

首先说明，不带惩罚项的线性回归进行可逆线性变换是不会影响预测值的。而上面的两种 scaling都是可逆线性变换，因此进不进行 scaling 不会影响预测值。

但是，这是否真能说明线性回归不需要scaling？

个人认为，**即使线性变换不会影响预测值，线性回归也需要进行scaling**。

这需要从scaling带来的好处来看。scaling带来的一个显著好处是消除不同变量量纲的影响。因此，虽然scaling不会影响预测值，但是会影响参数的估计值。

而从可解释性的角度来说，参数的估计值会影响到可解释性，对于线性回归这类比较简单的模型来说，其解释性是其一个很重要的优点，因此，线性回归也应该进行scaling。

而逻辑回归只是线性回归的估计值之上加一个 sigmoid 函数。由于估计值不变，因此套一个 sigmoid 函数也不会变。

所以，**个人认为逻辑回归也需要进行scaling**。

## 注意事项

一定要记得先拆分验证集，后再做 scaling。同时，不能 training set  和 testing set/ validation set 分别做归一化。而是应该在 training set 上做 scaling之后，用 training set 上的一些指标，如均值、方差等在 testing set / validation set 上进行归一化。

常见的错误：
1. 在整个数据集上做scaling，后再划分 testing set / validation set，这样会将 testing set / validation set 中的信息引入 training set 中。
2. 先划分了 testing set / validation set，但是是 training set 和 validation set 分开归一化的。这样会导致一个问题：如果 testing set 中只有一个数据要如何进行归一化？还是统一使用 training set 上的指标进行归一化。

# Appendix

## 证明: 可逆线性变换不会改变估计值

假设样本为 $\{x_i, y_i\}_{i=1}^N$ ，其中 $x_i$ 是 p 维的行向量。

将  $x_i$ 拼成一个  $N \times (p+1)$ 的数据阵 $X$ ，其中第一列为全1向量，相当于将截距项放到了数据阵  $X$ 中。同时将 $y_i$ 拼成一个 $N$ 维向量  $y$ 

假设回归方程为 

$$
y = X \beta  + \epsilon
$$ 
则参数估计为

$$
\hat{\beta} = (X^TX)^{-1}X^{T} y
$$ 

对于一个新的样本  $(x_0, y_0)$ ，$y_0$ 的估计值为  

$$
\hat{y}_0 = x_0 \hat{\beta} = x_0(X^TX)^{-1}X^{T} y
$$ 

对 $X$ 进行线性变换，设线性变换的可逆方阵为  $A_{( p+1 ) \times (p+1)}$ ，则变换后的矩阵为 

$$
\tilde{X} = XA
$$ 

则变换后的参数估计值为

$$
\begin{aligned}
    \tilde{\beta } &= (\tilde{X}^T\tilde{X})^{-1}\tilde{X}^{T} y \\
    &= (A^TX^TXA)^{-1}A^TX^{T} y \\
    &= A^{-1}(X^TX)^{-1}(A^T)^{-1}A^TX^{T} y \\
    &= A^{-1}(X^TX)^{-1}X^{T} y \\
\end{aligned}
$$ 

对于新样本 $(x_0, y_0)$ 来说，$x_0$ 也需要进行线性变换，变成  $\tilde{x}_0 = x_0 A$ ，因此新的预测值

$$
\begin{aligned}
    \tilde{y}_0 &= \tilde{x}_0 \tilde{\beta} \\
    &= x_0 A A^{-1}(X^TX)^{-1}X^{T} y \\
    &= x_0 (X^TX)^{-1}X^{T} y \\
    &= \hat{y}_0 \\
\end{aligned}
$$ 

因此，对于线性回归来说，进行 scaling 不会改变预测值。

# References
1. [归一化与scaling - 无影随想 - 博客园](https://www.cnblogs.com/zhaokui/p/5112287.html)
2. [数据特征:scaling和归一化_人工智能_付石头的博客-CSDN博客](https://blog.csdn.net/u010947534/article/details/86632819)
3. [[译] 什么时候需要进行数据的scaling？为什么？ - 掘金](https://juejin.im/post/5d41a46bf265da03d727f85d)
4. 葫芦书
5. [(1 封私信 / 26 条消息) 数据预处理的归一化手段应该如何应用到训练集，测试集和验证集中？ - 知乎](https://www.zhihu.com/question/60490799/answer/214685372)
