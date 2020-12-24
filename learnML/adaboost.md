[toc]

# Job 学习笔记 5-Adaboost

## AdaBoost的计算过程

AdaBoost算法需要回答的问题有两个：

1. 如何改变样本的权重（也即如何调整**经验分布**）
2. 如何将分类器组合起来。

### 算法：AdaBoost

记 N 个样本 $\{(x_i, y_i)\}_{i=1}^N$

基学习器为 $G(x;\theta)$，$G(x; \theta) \mapsto \{-1, 1\}$； 基本学习器的数量为 $M$ ，即最终的分类器为 $M$ 个基学习器的和：

$$
\hat{G}(x) = sign \left( \sum_{m=1}^M \alpha_m G\left(x; \theta_m\right) \right)
$$

记第 $m$ 次迭代的经验分布为 $D_m = (w_{m1}, \dots, w_{mN})$，学习器为 $G(x; \theta_m)$，为了简化符号,记 $G(x;\theta_m)=G_m(x)$

算法描述如下：

1. 初始化经验分布为 $w_{1i}=\frac{1}{N}, i=1,\cdots,N$

2. 迭代 M 次，每次迭代都做下面三件事

    1. 计算在经验分布 $D_m = (w_{m1}, \dots, w_{mN})$ 下的学习器  $G_m(x)$

    2. 更新权重。其中又包含两个步骤：

        1. 计算 $G_m(x)$ 在经验分布 $D_m$ 下的**分类误差** $e_m$，其中
        $$e_m = \sum_{i=1}^N w_{mi} I(G_m(x_i) \neq y_i)$$ 

        2. 计算 $G_m(x)$ 的权重 $\alpha_m$，其中
        $$\alpha_m=\frac{1}{2} \log \frac{1 - e_m}{e_m}$$

        3. 更新经验分布 $D_{m+1} = (w_{m+1, 1}, \dots, w_{m+1, N})$，其中

        $$w_{m+1,i} = \frac{w_{mi} \exp (-y_i \alpha_m G_m(x_i))}{Z_m}$$
        其中 $Z_m$ 是归一化系数，即 $Z_m = \sum_{i=1}^{N}w_{mi} \exp (-y_i \alpha_m G_m(x_i))$

3. 输出最终的分类器 $\hat{G}(x) = sign \left(\sum_{m=1}^M \alpha_m G_m(x) \right)$


对于下面的AdaBoost算法来说，**第一个问题**是是通过降低分类错误的样本权重和增加分类正确的样本权重来调整经验分布的。

而**第二个问题**分类器的组合则是通过加权平均，将所有基本学习器加权平均起来得到的。对于那些比较弱的学习器（即分类错误率比较大的学习器），权重较小。

注意上面的表述中，有两个"权重"。一个是**样本的权重**，一个是**基本学习器的权重**。样本的权重相当于经验分布。Adaboost，将基本学习器的权重和经验分布的权重融合在了一起，在计算经验分布的权重过程中，会得到基本学习器的权重（即下文中的 $\alpha_m$）

## 加法模型与前向分步算法

对于分类器 $G_{m}(x;\theta)$ 将 $x$ 映射到  $\{1, -1\}$  ，考虑加法模型

$$
\begin{aligned}
    f(x) &= \sum_{m=1}^{M} \alpha_m G_m(x; \theta_m) \\
    &=  \sum_{m=1}^{M} \alpha_m G_{m}(x) 
\end{aligned}
$$ 

损失函数 

$$
L = \frac{1}{N} \sum_{i=1}^{N} L(y_{i}, f(x_i)) = \frac{1}{N} \sum_{i=1}^{N} L(y_i, \sum_{m=1}^{M} \alpha_mG_m(x_i))
$$ 

上面的损失函数有 $2m$  个参数$(\alpha_i, G_i(x)), i=1, \ldots, m$ ，其中 $G_i(x)$ 是一个函数，我们也将其看做一个参数。 

因为不好一次性优化，前向分步算法就利用贪心的思想，分步优化上面的损失

1. 先优化 $(\alpha_1, G_{1}(x))$，即优化 $\frac{1}{N} \sum_{i=1}^{N} L(y_i, \alpha_1 G_1(x_i)$

2. 再优化 $(\alpha_2, G_2(x))$ （此时  $(\alpha_1,G_1(x))$ 看作已知），即优化 $\frac{1}{N} \sum_{i=1}^{N} L(y_i, \alpha_1G_1(x_i) + \alpha_2G_2(x_i))$

3. 一直往后优化下去，直到求出 $(\alpha_m, G_m(x))$ 。

### 算法：前向分步算法

1. 初始化 $f_0(x)=0$ 

2. for m = 1 to M do

    1. $(\alpha_m, G_m(x)) = \mathop{\arg\min_{(\alpha, G(x))}} L(y_i, f_{m-1}(x_i) +\alpha_m G_m(x_i))$ 

    2. $f_{m}(x) = f_{m-1}(x) + \alpha_m G_m(x)$ 

3. 输出 $G(x) = sign \left(\sum_{m=1}^{M} G_m(x) \right)$

### 利用前向分步算法推导 AdaBoost

上面的前向分步算法中，第2步的优化只是形式化的表示，如果取 $L(y, f(x)) = \exp(-yf(x))$ 。代入第2步中，有

$$
\begin{aligned}
    L &= \sum_{i=1}^{N} \exp \left( -y_i (f_{m-1}(x_i) + \alpha_m G_m(x))\right) \\
    &= \sum_{i=1}^{N} \exp (-y_i f_{m-1}(x_i)) \exp(-y_i \alpha_m G_m(x_i)) \\
    &= \sum_{i=1}^{N} \overline{w}_{mi} (-y_i \alpha_m G_m(x_i))
\end{aligned}
$$ 

其中 $\overline{w}_{mi} = \exp(-y_i f_{m-1}(x_i))$ 

#### 求 G_m(x) 

先固定 $\alpha_m$ 优化 $G_m$ 

$$
\begin{aligned}
    L &= \sum_{y_i= G_m(x_i)} \overline{w}_{mi} \exp(-\alpha_m) + \sum_{y_i \neq G_m(x_i)} \overline{w}_{mi} \exp(\alpha_m) \\
    &= \sum_{i=1}^{N} \overline{w}_{mi} \exp(-\alpha_m) -
    \sum_{y_i \neq G_m(x_i)} \overline{w}_{mi} \exp(-\alpha_m) +
    \sum_{y_i \neq G_m(x_i)} \overline{w}_{mi} \exp(\alpha_m) \\
    &= \sum_{i=1}^{N} \overline{w}_{mi} \exp(-\alpha_m) +
    \sum_{y_i \neq G_m(x_i)} \overline{w}_{mi} (\exp(\alpha_m)  - \exp(-\alpha_m)) \\
    &= \sum_{i=1}^{N} \overline{w}_{mi} \exp(-\alpha_m) + (\exp(\alpha_m)  - \exp(-\alpha_m)) err_{m} \\
\end{aligned}
$$ 

上式与 $G_m(x)$ 有关的只有  $err_m$ ，而 $(\exp(\alpha_m)  - \exp(-\alpha_m))$ 为正，因此 $err_m$ 越小 $L$ 越小。

所以 $G_m(x)$ 需要使  $err_m$ ，而  $err_m$ 是调整权重后的错误率，而让 $err_m$ 最小的  $G_m(x)$ 只需要将样本调整权重后使用基学习器训练一个模型。

#### 求 \alpha_m

固定 $G_m(x)$，对 $L$ 做如下推导

$$
\begin{aligned}
    L &= \sum_{y_i= G_m(x_i)} \overline{w}_{mi} \exp(-\alpha) 
    + \sum_{y_i \neq G_m(x_i)} \overline{w}_{mi} \exp(\alpha) \\
    &= (1-err_m)\exp(-\alpha) + err_m \exp(\alpha) \\
    & \ge \sqrt{2(1-err_m)err_m}  
\end{aligned}
$$ 

最后一步是根据均值不等式得到，当且仅当 $\exp(-\alpha)(1-err_m) = \exp(\alpha) err_m$ 时等号成立，即

$$
\alpha_m = 2 \log \frac{1-err_m}{err_m}
$$ 

# References

1. 李航统计学习方法
2. [Friedman , Hastie , Tibshirani : Additive logistic regression: a statistical view of boosting (With discussion and a rejoinder by the authors)](https://projecteuclid.org/euclid.aos/1016218223)


