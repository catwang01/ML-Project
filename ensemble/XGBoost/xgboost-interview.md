[toc]

# XGBoost面试题

## 1. XGBoost和GBDT有什么不同

1. 基分类器： 
    - >XGBoost 支持 CART树，也支持线性分类器，此时 XGBoost 相当于带 L_1 和 L_2 正则化的 Linear Regression(for regression) 或 Logistic Regression(for classification) 
    - 这段是网上抄的，但是感觉说的不是很有道理。实际上，个人认为它们的基分类器最大的区别应该是目标函数的不同。
    - GBDT 只支持 CART树。

2. 导数信息
    - XGBoost用到了二阶信息
    - GBDT只用到了一阶信息

3. 正则项
    - XGBoost使用了正则项。由两部分组成，一部分是对叶结点数量做出的惩罚，另一部分是对每个叶结点上的输出分数的惩罚。
    - GBDT 没有使用正则项

4. 列采样
    - XGBoost使用了类似于 RandomForest 的列采样以减少过拟合
    - GBDT没有使用

5. 缺失值处理
    - XGBoost有缺失值处理。会尝试将缺失值分到左右的结点上去并选择使用损失减少最多的那个作为缺失值的分类。
    - GBDT 没有缺失值处理

6. 并行化
    - XGBoost可以在**特征级别**上进行并行化
    - GBDT不行。（其实个人认为 GBDT 也可以在特征级别上进行并行化，只需要借鉴 XGBoost 的那部分就可以了）

## 2. XGBoost为什么可以并行

XGBoost的并行不是每棵树进行并行，它本质上还是 Boosting 方法，所以不能完成每个基分类器的并行训练。

XGBoost的并行指的是特征粒度的并行，不是tree粒度的并行。XGBoost在选取最佳分割点的时候不同的特征可以分别计算，最后选择最优的那个特征来进行分割。

## 3. XGBoost为什么快

因为XGBoost做了许多优化以适应大规模数据集的训练。

1. 分桶：XGBoost在选择划分点时使用的是贪心算法。对于比较大规模的数据集来说，XGBoost可以使用Approximate Greedy Algorithm 来实行计算。通过分桶的方式的方式来样大数据的颗粒度，以加速计算。

2. block 中的数据只需要进行一次排序，之后都使用之前的排序结果，因此可以减少排序。

3. 列采样。XGBoost中使用了 Random Forest 中的列采样，在防止过拟合的同时，也可以减少计算量（因为可以少算一些特征）

4. 对特征粒度进行并行。 

5. cache aware accesss。XGBoost在为了避免cache miss使用了数据预取的计算。对于 Approximate Greedy Algorithm 来说，还可以调整合适的 blocksize 以减少 cache miss

## 4. XGBoost如何防止过拟合

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

2. 分割特征的选取采用了类型于 RandomForest 中的列采样计算，随机选取一些特征作为候选特征，而非使用全部特征。

3. 具体的选取算法有 Exact Greedy Algorithm 和 Approximate Greedy Algorithm。而 Approximate Greedy Algorithm 需要对样本进行分桶。分桶是通过加权二阶导数来进行的。Approximate Greedy Algorithm 还有 global 和 local 划分的区别。 

    global 是指在最开始给出候选分割点之后，接下来都使用这些候选分割点去计算。而 local 是最每次分裂都先计算出新的候选分割点，再在这堆新的候选分割点中选择最好的那个。可以看到，local 的计算量更大一些，效果也相对比较好一些。

## 9. 为什么XGBoost相比某些模型对缺失值不敏感

实际上这个是所有树模型都具有的一个特性。

因为树模型在结点划分的时候是寻找最优的分裂点，这个分裂点的寻找不怎么会受到缺失值的影响，因此树模型对缺失值不敏感。

而且，在这个基础上，XGBoost还有专门处理缺失值的方法。

## 10. XGBoost 的重要性是如何排序的？

xgboost 可以选择多种重要性计算方式。有下面几种方式：

1. weight 特征被选做分裂点的次数
2. total_gain 每个特征被选做分裂点之后会带来增益，每个特征可能多次被当作分裂点，会多次带来增益。将这些增益加起来代表特征的重要程度。
3. gain: total_gain / weight。总增益 / 被当作分裂点的次数
4. total_cover: 这个特征分裂了的样本的个数的总和。
5. cover： total_cover / weight

## 11. 为什么 random forest 的树要比 gdbt / XGBoost 的要深？


因为 random forest 是 bagging 方法。bagging 方法致力于降低方差而非偏差。因此，random forest 的基分类器的拟合能力要足够强，才能保证模型的偏差比较小。而决策树越深，模型的偏差越小。 

而 gdbt / XGBoost 是 boosting 方法。boosting 方法致力于减少偏差而非方差。因此，gdbt 的基分类器的拟合能力如果太强， 反倒会导致过拟合的风险。因此，random forest 的决策树要比 gdbt / XGBoost 的树深度大。random forest 深度大是为了减少偏差，从而保证拟合能力。而 gdbt / XGBoost 的树深度小是为了减少过拟合的风险。

## 12. GDBT 如何处理多分类任务

GDBT 通过训练 K 个分类器的方法来处理多分类任务。大概有下面几个点：
1. 同时训练 K 个分类器。假设一共迭代 n 次（即每个分类器有 n 课树）。那么一共会有 n K 棵树。
2. 每次迭代，都拟合 K 个分类器。不能每个类别分开训练。
3. K 类的 target 中一个为1，其他为0。

## 13. RF 的随机性体现在什么地方？

RF 的随机性体现在
1. 训练样本是随机的，每棵树的训练样本是原始样本的一个 boostrap 抽样。
2. 树的分裂节点的候选集合也是随机的。

## 14. gdbt / XGBoost 如何处理类别变量？

由于 gdbt / XGBoost 分裂的过程将所有变量当作连续型变量，因此对于类别变量，有两种处理方式：
1. 对于有序的类别变量，如年级、年龄等，可以使用有序编码。如将 [ 幼年，中年，老年 ] 编码成 [0, 1, 2]
2. 对于无序的类别变量， 如性别，使用 one-hot 编码。对于 gdbt / XGBoost 来说，相当于增加了树的深度。（如果某个变量有10个类比，那么就需要分裂10次）

## 15. 为什么XGBoost不使用处理高维稀疏问题？而 LR 更加适合？

这个问题实际上也是树模型的一个特性。

树模型不使用处理高维稀疏问题的原因有两个：
1. 计算开销上看：分割时需要考虑所有的特征。因此特征越多，计算开销越大。
2. 从过拟合的角度看：其次，在高维稀疏问题上更容易过拟合，因为增加特征实际上是在增加树的深度。

LR的目标就是找到一个超平面对样本是的正负样本位于两侧，由于这个模型够简单，不会出现gbdt上过拟合的问题，不过它可能要但心欠拟合的问题了。

讲真从这个角度来看，XGBoost算是树模型中处理高维稀疏问题比较好的了。一方面有列采样减少特征分裂中的时间。另一方面还对每个叶结点的输出分数有惩罚。


# References
1. [珍藏版 | 20道XGBoost面试题 - 云+社区 - 腾讯云](https://cloud.tencent.com/developer/article/1500914)
2. [CTR 预测理论（十九）：高维稀疏特征场景中 LR 比 GBDT 效果好的原因_CTR 预测理论（十九）,高维稀疏特征场景,LR_Dby_freedom的博客-CSDN博客](https://blog.csdn.net/Dby_freedom/article/details/98658805?depth_1-utm_source=distribute.pc_relevant.none-task-blog-OPENSEARCH-1&utm_source=distribute.pc_relevant.none-task-blog-OPENSEARCH-1) 
