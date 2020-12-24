[toc]

# Todo - 2020-05-08 09:50 -- by ed  这个之后有时间再看

# ML 学习笔记 4-评价标准

## 4.1 回归问题的评价标准

对于回归问题，最简单的就可以取 loss function 的值作为评价标准。比如对于最简单的线性回归问题，其 loss function 为平方损失函数

$$
loss = \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$ 

对可以取这个值来作为评价标准。

## 4.2 分类问题的评价标准

对于分类问题来说，评价标准就比较复杂了。

先给出混淆矩阵。下面的介绍很多是基于混淆矩阵的。

![confusion matrix ](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200428111323.png)

其中，第一个字母 $T$, $F$  表示的是是否分类正确。第二个字符  $P$, $N$  表示的是预测的类别。

### 4.2.1 ACC

最常用的评价标准是准确率，即预测正确的比率

$$
acc = \frac{TP + TN}{TP + FP + TN + FN}
$$ 

其中分子 $TP + FP + TN + FN$ 表示的是所有的样本个数。

可以看到，这个指标综合考虑了正负样本的分类正误情况，并且给予正负样本以相同的权重。但是，如果出现数据不平衡的情况，比如正样本数远大于负样本数，即

$$
TP + FN >> FP + TN
$$ 

则 acc 的计算中分子主要由 $TP$ 确定，而分母主要由 $TP + FN$  确定。因此此时 acc 主要衡量了正例被分正确的程度，而不怎么考虑负例被分类正确的程度。

如果我们关心的是负类被分类正确的程度的话，使用 acc 来度量效果就不会很好。

>首先，accuracy是最常见也是最基本的evaluation metric。但在binary classification 且正反例不平衡的情况下，尤其是我们对minority class 更感兴趣的时候，accuracy评价基本没有参考价值。什么fraud detection（欺诈检测），癌症检测，都符合这种情况。
来自于 [2]

### 4.2.2 Precision & Recall 

对于一些问题，我们会关心的查全率（recall）。其定义为

$$
recall = \frac{TP}{ TP + FN }
$$ 

相对的，我们叫之前的acc起一个新的名字叫 precision。

$$
precision = \frac{TP}{TP + FP}
$$ 

![picture](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200428112626.png)

下面的是个人理解：
这个问题实际上涉及到正例与负例的不平衡性。不平衡性有两个维度，一个是数量上的，一个是代价上的。

- 数量上，就是正例与负例的样本个数不平衡
- 代价上，就是错判为正例即 FP 和错判为负例 FN 的代价不同。如医学中的病毒检测，将一个有病的人判断为没病的代价，要大于将一个没病的人判断有病的代价。因此这种问题上关注的点就不是准确率，而是查全率。这有点类似于统计中的原假设和备择假设的关系。

注意到，recall 和 precision 是一种 tradeoff 的关系。极端点来说，如果希望recall较大，将可以将所有的样本都预测为正例，此时 recall=1，
但precision会低，因为将所有的负例都标记为正了；

如果希望precision较大，可以只将最有把握的样本标记为正，其实都标记为负，这样会漏掉许多正例样本，使得 recall 偏低。

#### F1

我们可以对这两个指标取平均值得到一个综合考虑这两个指标的一个新的指标 F1，其定义为

$$
\frac{1}{F1} = \frac{1}{2} (\frac{1}{precision} + \frac{1}{recall})
$$ 

这里取的是 precision 和 recall 的调和平均值。

还可以将 F1 的概念推广，取 precision 和 recall 的加权调和平均值，这里就不介绍了。

F1 的缺点：
1. 一个 precision 大而 recall 小和一个 precision 小或 recall 大的模型的  f1 可能是相同的，因此也不适合处理我们更多关注 minority data 的情况。

- 多分类情况

对于多分类情况，的F1计算有两种方式。其实很无聊，没有什么意思，就是先计算平均再计算 precision 和 recall 还是先计算precision 和 recall 再求平均的区别

## 4.3 ROC和AUC

对于样本不平衡的情况，我们发现正例与负例的重要程度是不同的。那么，划分正例与负例的阈值是不是也会受到影响？通常我们取0.5作为阈值，这个阈值是否仍然恰当？

这个问题可以通过ROC曲线来回答。

ROC曲线的横坐标是假例率 FPR（flase positive rate），而纵坐标是正例率TPR（true positive rate），其中这两个指标的定义为 

$$
\begin{aligned}
TPR = \frac{TP}{TP + FN} \\
FPR = \frac{FP}{FP + TN}
\end{aligned}
$$ 

可以看到，TPR实际上就是之前定义的 recall

![image](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200428112659.png)

- ROC曲线的绘制

ROC曲线的绘制是通过改变分类的阈值，每个阈值都对应坐标轴上的一个点，将所有的点都绘制出来就得到了ROC曲线。

- ROC曲线的作用

ROC曲线主要有两个用途
1. 判断最好的阈值
2. 判断两个模型的好坏。如果一个模型的ROC曲线包住了另一个模型的，说明这个模型更好。但是，实际上很少有曲线可以完全包住另一个曲线，更常见的情况是两个曲线有相交，这时，可以计算ROC曲线下的面积，即 **AUC**（area under ROC curve）。AUC较大的曲线对应的模型更优。

# References
1. 西瓜书
2. [精确率、召回率、F1 值、ROC、AUC 各自的优缺点是什么？ - 知乎](https://www.zhihu.com/question/30643044)
