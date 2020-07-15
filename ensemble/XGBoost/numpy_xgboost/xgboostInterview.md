[toc]

# XGBoost面试题

## 1. XGBoost和GDBT有什么不同

1. 基分类器： - XGBoost支持 CART树，也支持线性分类器，此时 XGBoost 相当于带 L_1 和 L_2 正则化的 Linear Regression(for regression) 或 Logistic Regression(for classification)
    - GDBT 只支持 CART树。
2. 导数信息
    - XGBoost用到了二阶信息
    - GDBT只用到了一阶信息
3. 正则项
    - XGBoost使用了正则项。由两部分组成，一部分是对叶结点做出的惩罚，另一部分是对每个叶结点上的输出分数的惩罚。
    - GDBT 没有使用正则项

4. 列采样
    - XGBoost使用了类似于 RandomForest 的列采样以减少过拟合
    - GDBT没有使用

5. 缺失值处理
    - XGBoost有缺失值处理。会尝试将缺失值分到左右的结点上去并选择使用损失减少最多的那个作为缺失值的分类。
    - GDBT 没有缺失值处理

6. 并行化
    - XGBoost可以在特征级别上进行并行化
    - GDBT不行。

## 2. XGBoost为什么可以并行

XGBoost的并行不同每棵树进行并行，它本质上还是 Boosting 方法，所以不能完成每个基分类器的并行训练。

XGBoost的并行指的是特征粒度的并行，不是tree粒度的并行。XGBoost在选取最佳分割点的时候不同的特征可以分别计算，最后选择最优的那个特征来进行分割。

## 3. XGBoost为什么快

因为XGBoost做了许多优化以适应大规模数据集的训练。

1. 分桶：XGBoost在选择划分点时使用的是贪心算法。对于比较大规模的数据集来说，XGBoost可以使用Approximate Greedy Algorithm 来实行计算。通过分桶的方式的方式来样大数据的颗粒度，以加速计算。

2. block 中的数据只需要进行一次排序，之后都使用之前的排序结果，因此可以减少排序。

3. 列采样。XGBoost中使用了 Random Forest 中的列采样，在防止过拟合的同时，也可以减少计算量（因为可以少算一些特征）

4. 通过将数据划分为 block，XGBoost可以实现 block 级别的并行，提高效率。

5. cache aware accesss。XGBoost在为了避免cache miss使用了数据预取的计算。对于 Approximate Greedy Algorithm 来说，还可以调整合适的 blocksize 以减少 cache miss

6. out of core computation
    - block 的数据压缩。对列进行数据压缩。对行使用16位 int 来存储，并且只存储 index 的 offset，可以减少数据大小。.
    - block sharding 将 block 中的数据保存在不同的磁盘中，以提高吞吐。

## 4. XGBoost防止过拟合

1. 目标函数添加正则项。正则项分为两个部分，一个是对叶结点数量的惩罚，一个是对每个叶结点上输出分数的惩罚。
2. 列采样，即可以防止过拟合，还可以减少需要计算分裂点的特征数量。
3. shrinkage，采用下面的式子来更新

$$
\hat{y}^{(t)}  = \hat{y}^{(t-1)} + \eta f_t
$$ 

## 5. XGBoost的推导

## 6. Todo - 2020-04-25 16:24 -- by ed XGBoost如何处理不平衡样本

## 7. Todo - 2020-04-25 16:27 -- by ed XGBoost中如何对树进行剪枝

1. 添加正则项本身就相当于预剪枝。

## 8. XGBoost如何选择最佳分割点？

1. 首先，选择最佳分割点的策略是增益最大

$$
L_{split} = \frac{1}{2}[\frac{G_L}{H_L+\lambda} + \frac{G_R}{H_R+\lambda}  - \frac{G}{H+\lambda}]  - \gamma
$$ 

2. 分割点的选取采用了类型于 RandomForest 中的列采样计算，随机选取一些特征作为候选特征，而非使用全部特征。

3. 具体的选取算法有 Exact Greedy Algorithm 和 Approximate Greedy Algorithm。而 Approximate Greedy Algorithm 需要对样本进行分桶。分桶是通过加权二阶导数来进行的。Approximate Greedy Algorithm 还有 global 和 local 划分的区别。 

    global 是指在最开始给出候选分割点之后，接下来都使用这些候选分割点去计算。而 local 是最每次分裂都先计算出新的候选分割点，再在这堆新的候选分割点中选择最好的那个。可以看到，local 的计算量更大一些，效果也相对比较好一些。

## 9. 为什么XGBoost相比某些模型对缺失值不敏感

实际上这个是所有树模型都具有的一个特性。

因为树模型在结点划分的时候是寻找最优的分裂点，这个分裂点的寻找不怎么会受到缺失值的影响，因此树模型对缺失值不敏感。

而且，在这个基础上，XGBoost还有专门处理缺失值的方法。

## Todo - 2020-04-25 17:55 -- by ed 10. 为什么XGBoost不使用处理高维稀疏问题？而 LR 更加适合？

这个问题实际上也是树模型的一个特性。

树模型不使用处理高维稀疏问题的原因有两个：
1. 分割时需要考虑所有的特征。因此特征越多，计算开销越大。
2. 其次，在高维稀疏问题上更容易过拟合。具体可以看 [2] 中的例子。

LR的目标就是找到一个超平面对样本是的正负样本位于两侧，由于这个模型够简单，不会出现gbdt上过拟合的问题，不过它可能要但心欠拟合的问题了。

讲真从这个角度来看，XGBoost算是树模型中处理高维稀疏问题比较好的了。一方面有列采样减少特征分裂中的时间。另一方面还对每个叶结点的输出分数有惩罚。


# References
1. [珍藏版 | 20道XGBoost面试题 - 云+社区 - 腾讯云](https://cloud.tencent.com/developer/article/1500914)
2. [CTR 预测理论（十九）：高维稀疏特征场景中 LR 比 GBDT 效果好的原因_CTR 预测理论（十九）,高维稀疏特征场景,LR_Dby_freedom的博客-CSDN博客](https://blog.csdn.net/Dby_freedom/article/details/98658805?depth_1-utm_source=distribute.pc_relevant.none-task-blog-OPENSEARCH-1&utm_source=distribute.pc_relevant.none-task-blog-OPENSEARCH-1) 
