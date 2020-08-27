[toc]

# Kmean 算法的常见面试题

## 1. 简述一下K-means算法的原理和工作流程
## 2. K-means中常用的到中心距离的度量有哪些？

1. 曼哈顿距离
2. 欧式距离
3. 余弦相似度

## 3. K-means中的k值如何选取?

Elbow

## 4. K-means算法中初始点的选择对最终结果有影响吗？

## 5. K-means聚类中每个类别中心的初始点如何选择？

1. 选取距离尽量远的K个样本点作为中心点

随机选取第一个样本$C_1$ 作为第一个中心点，遍历所有样本选取离 $C_1$ 最远的样本 $C_2$ 作为第二个中心点，以此类推，选出K个初始中心点

## K-means 的评价指标

1. SSE：总平方和
2. 轮廓系数：

## 6. K-means中空聚类的处理
## 7. K-means是否会一直陷入选择质心的循环停不下来？如何解决？

1. 设置迭代次数
2. 增大设定收敛判断的阈值

## 8. 如何快速收敛数据量超大的K-means？
## 9. K-means算法的优点和缺点是什么？

- K-Means的主要优点：
1. 原理简单，容易实现
2. 可解释度较强

- K-Means的主要缺点：
1. K值很难确定
2. 局部最优
3. 对噪音和异常点敏感（因为它们可以显著影响SSE）
4. 需样本存在均值（限定数据种类）
5. 聚类效果依赖于聚类中心的初始化
6. 对于非凸数据集或类别规模差异太大的数据效果不好

# References
1.  [聚类（二）——KMeans算法（工作面试常用面试题入手）_数据结构与算法_WangZixuan1111的博客-CSDN博客](https://blog.csdn.net/WangZixuan1111/article/details/98970139?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3)
2. [KMeans原理、调参及应用_Python_liangtingac的专栏-CSDN博客](https://blog.csdn.net/liangtingac/article/details/48270233)
