[toc]

# ML 学习笔记-7-XGBoost-2-个人对于 XGBoost 的一些思考

xgboost 的那篇 paper，融合了之前的许多思想，并且在工程实现上有许多优化点。结果 ed 在刚开始读的时候只停留于这些很细节的东西，而忽略了最本质的东西。

这次在准备面试时再次学习了 XGBoost ，有一些比较独到的感受，因此在这里记录一下。

## 牛顿法

在提到 XGBoost 的时候，大家都会说类似的话：“XGBoost 比 GBDT 好，因为 XGBoost 将泰勒公式展开到了二阶，使用到了二阶信息 balabala”

但是，却很少听到有人提起牛顿法这个词（不过文献 [ 2 ] 中有提到）。

ed 从一开始也不是很明白所谓的“将泰勒公式展开到二阶”是这个什么意思。

如果用 GBDT 的思维来考虑的话，GBDT 是拟合负梯度 $- \frac{\partial L(y, \hat{y}}{\partial \hat{y}} |_{\hat{y}=F_{m-1}(x)}$  ，那么 XGBoost 就应该是拟合 $- \left( \frac{\partial L(y, \hat{y}}{\partial \hat{y}} |_{\hat{y}=F_{m-1}(x)} + \frac{1}{2} \frac{\partial^2 L(y, \hat{y})}{\partial \hat{y}^2} |_{ \hat{y}=F_{m-1}(x)} \right)$ 了？

显然，在看到 XGBoost 的那堆公式时就知道不是这样。实际上，ed 认为，所谓的将 Taylor 公式展开到二阶实际上就是在使用牛顿法优化目标函数。原 paper 将目标函数展开到二阶然后求二次函数的最小值的作法，实际上也早就牛顿法的证明中被使用过，这部分可以参考 [ 2 ]。这里不多说，直接盗 [ 2 ] 中的一张图。

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200708224800.png)

## XGBoost 的 base learner

在没有拜读 XGBoost 的paper的时候，ed 就知道 XGBoost 是一个 boosting 方法，而 boosting 方法最重要的就是如何根据上一个 learner 的结果来学习下一个 learner。

其中，GBDT 是通过对上一个 learner 结果的残差进行学习的，这个很直观。而 XGBoost 的论文中，却没有类似的东西。反而是一堆公式，然后又是节点划分的算法之类的。ed 感到非常困惑，不是 boosting 算法法？为什么要费劲去推一个增益计算的公式？还搞出了两上节点划分的算法？

**实际上，ed 认为，这是 XGBoost 一个比较核心的内容：重新设计 base learner。**

把握一种决策树算法，可以从下面几个方面着手：
1. 如何划分样本空间，即如何将样本空间划分为不同的区域。这个问题又可以归结为两个子问题
    1. 分裂标准（即损失函数）是什么？
    2. 如何根据损失函数分裂结点？
2. 每个叶节点的上的输出是如何计算的。

对于CART树来说，
1. 如何划分样本空间
    1. 分裂的标准（或者说是损失函数）是 gini index （分类问题）或 mse（回归问题）
    2. 二分递归分裂。通过启发式的贪心算法，遍历所有特征的所有取值来找到最佳分裂特征和最佳分裂特征对应的分裂取值，进而将结点划分为左右结点的。
2. 对于某个节点，如果是分类问题，那么这个节点上的预测值就是这个节点对应的区域中的所有样本标签的 majority voting；如果是回归问题，就是这个节点对应的区域中的所有样本的 target 的平均值。

我们上面说到，XGBoost 主要贡献是设计出了一个新的 base learner。这个 base learner 也可以从这几个方面来：
1. 如何划分样本空间
    1. 分裂标准是 loss 。公式在原论文的 2.2 节进行了推导
    2. 如何根据分裂标准来分裂结点。有两个算法 extract greedy algorithm 和  approximate greedy algorithm。这个对应于原论文的 section 3 SPLIT FINDING ALGORITHMS
2. 模型在每个节点是如何输出值的：这个也是在原论文的 2.2 节点和分裂标准一起推导出来的。

下面对这几个部分进行详细解释

### 分裂的标准

#### 为什么要重新设计 base learner？

有一个关键的问题要解决：为什么要重新设计一个 base learner？有没有必要？CART 它不香吗？

对于这个问题，需要先看看 GBDT 用 CART 做 base learner 有什么问题。

首先，ed 认为，GBDT 中有两个训练的过程
1. 一个是 CART 的训练过程，这个过程以残差（负梯度）为目标值进行训练，这个部分的 loss 是 CART 的loss，即 gini index 或 mse，不妨将这个loss称为 local loss。
2. 另一个是如何组合当前的 base learner 的训练，这个部分的 loss 是 GBDT 设置的 loss，通常是 mse 或者是 crossentropy，不妨将其称为 global loss。

而我们学习的最终目标应该是 global loss 最小，但是第一个过程的训练过程实际上是以 local loss 而不是 global loss 为优化目标的，这就导致两个训练目标不一致。

#### XGBoost 的 base learner 的 loss

而 XGBoost 的一个改进就是将 local loss 变成了 global loss 的一部分。

对于 XGBoost，其 global loss 是

$$
\begin{array}{l}
\mathcal{L}(\phi)=\sum_{i} l\left(\hat{y}_{i}, y_{i}\right)+\sum_{k} \Omega\left(f_{k}\right) \\ \\
\text { where } \Omega(f)=\gamma T+\frac{1}{2} \lambda\|w\|^{2}
\end{array}
$$

而 XGBoost 的 base learner 的 local loss 是

$$
\tilde{\mathcal{L}}^{(t)}(q)=-\frac{1}{2} \sum_{j=1}^{T} \frac{\left(\sum_{i \in I_{j}} g_{i}\right)^{2}}{\sum_{i \in I_{j}} h_{i}+\lambda}+\gamma T
$$

这个实际上是 global loss 的一部分变形得到的。因此，XGBoost 的 base learner 的 local loss 和 global loss 是一致的，是 global loss 的一部分。


### 如何利用分裂标准进行分裂

XGBoost 的 base learner 和 CART 的分裂方式相同的，论文中的 exact greedy algorithm 实际上就是 CART 的结点分裂算法。而只不过，XGBoost 在这个基础上利用了分桶思想提出了一个 approximate greedy algorithm 而已。

个人认为这个不算 XGBoost 的本质，因为 CART 也完全可以使用所谓的 approximate greedy  algrithom 来分裂。

### 如何计算节点上的输出值

这个也是在 2.2 节推导出来的。这里直接给出公式

$$
w_{j}^{*}=-\frac{\sum_{i \in I_{j}} g_{i}}{\sum_{i \in I_{j}} h_{i}+\lambda}
$$


可以看到，这个式子上面是一队导数，下面是二阶导数，实际上就是牛顿法中的更新步长。





# References
1. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754v1.pdf)
2. 一个非常不错的参考资料 [http://wepon.me/files/gbdt.pdf]( http://wepon.me/files/gbdt.pdf )
