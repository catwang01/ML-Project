
[toc]

# ML 学习笔记 6-GDBT - 1

最近在看关于XGboost的内容。其中非常喜欢作为面试题的内容就是XGboost和GDBT的比较。因此需要先看看GDBT。李航的书里面的GDBT的内容细节不是很多，因此还是找找GDBT的出处。

读完之后还是很有收获的。


## 梯度下降与最速下降

- 什么是最速下降（Steepest Descent）？和梯度下降法是一个东西吗？

有关 GDBT 的文章，都会提到说 GDBT 是通过最速下降法来优化的。实际上，最速下降法就是梯度下降法，这个可以参数 wikipedia 中的解释

>Gradient descent is also known as steepest descent

## 从参数的梯度下降到函数的梯度下降

### 关于参数的梯度下降

这一个就是我们经常在用的梯度下降法，假设损失函数为$L(y, F(x)))$ 。如果这个函数  $F(x)$ 是参数是，那么就可以表示为 $F(x;\theta)$  


#### 算法：关于参数的梯度下降

我们通过梯度下降的方法优化这个参数。

1. step1: 给定一个参数的初始值，$\theta_0$  

2. step2: 对于 $m=1, \ldots, M$ ，更新参数

    1. 计算负梯度 $g_m$ 

    $$
    g_m = - \mathbb{E} [\nabla_{\theta} L(Y, F(x)) |_{\theta=\theta_{m-1}} | x]
    $$ 

    2. 计算最优的学习率 $\eta_m$  
    有一点需要注意的是，这里的梯度下降的学习率 $\eta$  不是固定的，而是和当前的梯度  $g_m$  是有关的。这个和我们在深度学习中常见的 sgd 有点小区别。$\eta_m$ 由下面的式子确定

    $$
    \eta_m = \mathop{\arg\min_{\eta}} \mathbb{E} [L(Y, F(X, \theta_{m-1} + \eta g_m))]
    $$ 

    3. 更新 $\theta_m$ 

    $$
    \theta_m = \theta_{m-1} + \eta_m  g_m
    $$ 

    3. 最后得到的参数 $\theta$  的估计

$$
\hat{\theta} = \theta_M = \theta_0 + \eta_1 g_1 + \eta_2 g_2 + \dots + \eta_M g_M
$$ 

上面的推导是总体版本的，而要得到经验版本只需要将取期望 $\mathbb{E}$ 改成对样本求平均 $\frac{1}{N} \sum_{i=1}^{N}$ 就可以了

### 关于函数的梯度下降

上面是 $F(x)$ 是参数模型的情形。如果  $F(x)$ 是非参数模型，就不能用上面的方式来求了。不过还是有解决方法的。

对于  $F(x)$ 是参数模型的情况，如果固定一个 x，则  $F(x)$ 是固定的，因此可以看做是一个“参数”。通过这种方式，我们可以将一个非参数模型  $F(x)$ 看做有许多个参数  $F(x)$ ，其中 $x \in domF$ 。

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200706150637.png)

我们希望最小化的是 $\mathbb{E}L(Y, F(x)))$ ，注意这里的期望是对联合分布 $(X,Y)$ 取期望。而当固定  $X=x$  时，联合分布也就变成了条件分布，即我们固定 $x$ 后，希望最小化的函数变成了 

$$
\mathbb{E} [L(Y, F(x))|x] \tag{1}
$$ 

注意，这里的期望 $\mathbb{E}[\cdot | x]$ 表示的是对  $Y|X=x$ 取的期望 ，即在 $X=x$ 的条件下对 $Y$ 求期望。 


因此，我们可以继续使用梯度下降法来优化上面的条件期望 

#### 算法：关于函数的梯度下降

1. 确定初始的 $F_0(x)$ ，这里使用一个最优的常数来初始化 ，即

$$
F_0(x) = \mathop{\arg\min_c} \mathbb{E} [L(Y, c)|x]
$$ 

2. 对于 $m=1, \ldots, M$ ，进行下面的优化

    1. 计算负梯度 $g_m(x)$ 
    $$
    g_m(x) = - \mathbb{E} [\nabla_{F(x)} L(Y, F(x)) |_{F(x)=F_{m-1}(x)} | x]
    $$

    2. 计算最优学习率 $\eta_m$ 

    $$
    \eta_m = \mathop{\arg\min_{\eta}} \mathbb{E} [L(Y, F_{m-1}(x) + \eta g_m(x))|x]
    $$ 

    3. 更新 $F_{m}(x)$ 

    $$
    F_m(x) = F_{m-1}(x) + \eta_m g_t(x)
    $$ 

3. 输出

$$
\hat{F}(x) =F_{M(x)} = F_0(x) + \eta_1 g_1(x) + \eta_2 g_2(x) + \dots + \eta_M g_M(x)
$$ 

### 能否推广到样本场合？

上面的推导是基于总体场合的。能否将其推广到样本情形呢？

对于某些特殊的损失函数是可以的。如指数损失函数，在这种情况，上面的推导可以直接推出 Adaboost。（这个结果放在附录中）

但是，对于一般的损失函数，上面的推导是难以推广到样本形式的。原因有两个：

1. 因为难以计算 $E(\cdot|x)$ 。

2. 就算可以计算 $E(\cdot | x)$ ，也没有办法来直接得到 $g_m(x)$ 。因为 $g_m(x)$ 是一个函数。 

    我们在更新时实际上是用一个函数 $g_m(x)$ 来更新另一个函数 $F_{m-1}(x)$ ，这在总体情况下是可以做到的。但是在样本情形下，我们没有所有的 $x$，我们只有一些样本点。所以我们无法利用这些样本点来**直接**得到一函数 $g_m(x)$ 

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200422140440.png)

## Gradient Boost 拟合负梯度

我们虽然无法使用样本点来**直接**得到这个函数，但是我们可以**间接**得到这个函数。相当于将问题转化为：有一个函数 $g_m(x)$ ，需要用一堆样本点来表示这个函数，要怎么办？

当然是将  $g_m(x)$ 当作 $y$ 来进行拟合了！我们使用基函数 $h(x;a)$ 来拟合 $g_m(x)$ 。

### 算法：Gradient Boost

上面的算法可以推广到样本形式，得到了下面的 Gradient Boost 算法。

因此，**可以认为 Gradient Boost 算法是函数梯度下降在样本形式的一种近似处理**

1. 确定初始值 $F_0(x)$ ，这里还是使用一个最优的常数来进行初始化 
$$
F_0(x) = \mathop{\arg\min_c} \frac{1}{N} \sum_{i=1}^{N} L(y_i, c)
$$ 

2. 对于 $m=1, \ldots, M$ 进行下面的步骤

    1. 计算梯度在样本点处的值 $\{r_{mi}\}_{i=1}^N$ 
    $$
    r_{mi} =  - \frac{1}{n} \sum_i \nabla_{F(x)} L(y_i, F(x_i)) |_{F(x)=F_{m-1}(x)}
    $$ 

    2. 将 $\{r_{mi}\}_{i=1}^N$ 当作 $y$ ，用基函数来拟合，得到估计值 $\hat{h} = h(x; \hat{a}_m)$ 

    3. 计算最优学习率 $\eta_m$ 
     $$
    \eta_m = \mathop{\arg\min_{\eta}} \frac{1}{N} \sum_{i=1}^{N} L(y_i, F_{m-1}(x) + \eta h(x; \hat{a}_m))
    $$ 

    4. 更新函数 $F_m(x)$ 
    $$
    F_m(x) = F_{m-1}(x) + \eta_m h(x; \hat{a}_m)
    $$ 

3. 输出 $F_M(x)$

$$
F_M(x) = F_0(x) +  \eta_1 h(x; \hat{a}_1) + \dots + \eta_M h(x; \hat{a}_M)
$$ 

### 从加法模型和前向分步算法看Gradient Boost

上面是从梯度下降的角度来考虑，说明了 Gradient Boost 实际上是一种梯度下降算法，只不过对于函数。

另一方面，还可以从加法模型和前向分步算法的角度来考虑 Gradient Boost。
 
从加法模型和前向分步算法的角度来考虑，每一步都用基函数训练一个模型，使得当前的损失下降最多。而使当前损失下降最多的方向正好是负梯度的方向，因此只需要用基函数来拟合负梯度。

## GDBT

**上面说的是 Gradient Boost 算法。而 GDBT 是 Gradient Boost 算法在树模型中的应用**。

GDBT 就只是将上面的基学习器 $h(x;a)$ 选成 CART 树，然后在这个基础上做一些小变化而已。

CART树将样本空间划分成若干个区域 $R_1, \ldots, R_m$ ，每个区域对应一个预测值 $\alpha_i$ ，因此一棵 CART树可以表示为 

$$
h(x) = \sum_{i=1}^{N} \alpha_i I\{x \in R_i\}
$$ 

所以，当在第m步我们对于负梯度拟合出来一个CART树

$$
h_m(x) =  \sum_{i=1}^{N} \alpha_{mi} I\{x \in R_{mi}\}
$$ 

对应的最优学习率 $\eta_m$ 为

$$
\eta_m = \mathop{\arg\min_{\eta}} \sum_{i=1}^{N} L(y_i, F_{m-1}(x) + \eta h(x))
$$ 

更新函数 

$$
\begin{aligned}
    F_{m}(x) & = F_{m-1}(x) + \eta_m h_m(x) \\
    & = F_{m-1}(x) + \eta_m \sum_{i=1}^{N} \alpha_{mi} I\{x \in R_{mi}\} \\ 
    & = F_{m-1}(x) + \sum_{i=1}^{N} \eta_m \alpha_{mi} I\{x \in R_{mi}\}
\end{aligned}
$$ 


在上式中令 $c_{mi} = \eta_m \alpha_{mi}$ ，这相当于将学习率吸收到每个节点的预测值中去了，因此最后累加各棵树的结果时不需要计算学习率。

由此，我们可以得到下面的算法（这个算法来自 [4] ）

#### 算法：GDBT

1. 确定初始值 $F_0(x)$ ，这里还是使用一个最优的常数来进行初始化 
$$
F_0(x) = \mathop{\arg\min_c} \frac{1}{N} \sum_{i=1}^{N} L(y_i, c)
$$ 

2. 对于 $m=1, \ldots, M$ 进行下面的步骤

    1. 计算梯度在样本点处的值 $\{r_{mi}\}_{i=1}^N$ 

    $$
    r_{mi} =  - \frac{1}{n} \sum_i \nabla_{F(x)} L(y_i, F(x_i)) |_{F(x)=F_{m-1}(x)}
    $$ 

    2. 将 $\{r_{mi}\}_{i=1}^N$ 当作 $y$ 用基函数来拟合，得到估计值 $\hat{h} = h(x; \hat{a}_m)$ ，其叶结点的区域为 $R_{mj}, j=1, \dots, J$ 。注意到，如果将一个 CART 回归树看成两个部分：样本空间的划分和在每个区域上的取值的话，这一步只需要得到样本空间的划分即可。在每个区域上的取值和最优学习率结合起来在下一步进行优化。

    3. 计算每个区域上的最优评分

    $$
    c_{mj} = \mathop{\arg\min_{c}} \sum_{x_i \in R_{mj}} L(y_i, F_{m-1}(x_i) + c)
    $$ 

    这个实际上蕴含了对最优学习率的优化。

    4. 更新函数 $F_m(x)$ 

    $$
    F_m(x) = F_{m-1}(x) + \sum_{j=1}^{J} c_{mj} I\{x \in R_{mj} \}
    $$ 

3. 输出 $F_M(x)$


# Appendix

## 梯度下降推导 Adaboost

上面说过，因为函数的梯度下降无法处理样本的情况，因此才会需要 Gradient Boost。

Gradient Boost 可以看做是一种近似的梯度下降。

而对于一些特殊的情况，不需要做出这种近似就可以直接得到结果。Adaboost 就是一种特殊情况。

取损失函数为指数损失 $L\left( x \right) = e^{-yf\left( x \right)}$ 

利用 GB 算法进行计算，关键是计算负梯度 

$$
\begin{aligned}
    g_m(x) &= - \mathbb{E} [\nabla_{F(x)} L(Y, F(x)) | x] \\
    &= - \mathbb{E} [\nabla_{F(x)} e^{-yF\left( x \right) }| x] \\
    &= - \mathbb{E} [ -ye^{-yF\left( x \right) }| x] \\
    &= \mathbb{E} [ye^{-yF\left( x \right) }| x] \\
\end{aligned}
$$ 

$$
\mathbb{E_w} [g\left( x,y \right)  | x ] =  \frac{\mathbb{E}[w\left( x, y \right)g(x,y)|x]}{\mathbb{E}[w\left( x, y \right)|x]}
$$

将上面的 $e^{-yF(x)}$ 看做是权重 $w(x,y)$ 。因此上面的条件期望可以看做是一个加权后的条件期望，即

$$
g_m(x) = \mathbb{E}_w [y|x]
$$ 

这个结果实际上通过基函数 h 是对权重调整的分布进行拟合。即 $g_m(x)$ 通过基函数对权重调整后的分布进行拟合得到。

下面再计算最优学习率  

$$
\begin{aligned}
    & \mathbb{E}[L(Y, F(x) + \eta g_m(x))| x]  \\
    &= \mathbb{E}[e^{- Y (F(x) + \eta g_m(x))}| x] \\
    &= \mathbb{E}[e^{- YF(x) - Y \eta g_m(x)}| x] \\ 
    &= \mathbb{E_w}[e^{- Y \eta g_m(x)}| x] \\ 
    &= \mathbb{E_w}[e^{- Y \eta g_m(x)}(I\{Y=g_m(x)\} + I\{Y \neq g_m(x)\}) | x] \\
    &= \mathbb{E_w}[e^{- Y \eta g_m(x)}I\{Y=g_m(x)\}| x] + \mathbb{E_w}[e^{- Y \eta g_m(x)}I\{Y \neq g_m(x)\} | x] \\ 
\end{aligned}
$$

由于 $Y \in \{-1, 1\}$ ，$g_m(x) \in \{-1, 1\}$ ，因此在 ${Y=g_m(x)}$ 的区域上，有 $Yg_m(x) = 1$ ；在 ${Y \neq g_m(x)}$  的区域上，有  $Yg_m(x)=-1$ 

所以上式可以继续化简

$$
\begin{aligned}
    & \mathbb{E}[L(Y, F(x) + \eta g_m(x))| x]  \\
    &= \mathbb{E_w}[e^{- Y \eta g_m(x)}I\{Y=g_m(x)\}| x] + \mathbb{E_w}[e^{- Y \eta g_m(x)}I\{Y \neq g_m(x)\} | x] \\ 
    &= \mathbb{E_w}[e^{- \eta }I\{Y=g_m(x)\}| x] + \mathbb{E_w}[e^{\eta}I\{Y \neq g_m(x)\} | x] \\ 
    &= e^{- \eta }\mathbb{E_w}[I\{Y=g_m(x)\}| x] + e^{\eta}\mathbb{E_w}[I\{Y \neq g_m(x)\} | x] \\ 
\end{aligned}
$$ 

注意到 $\mathbb{E_w}[I\{Y=g_m(x)\}| x]$ 实际上是 $g_m(x)$ 在调整权重的分布上的错误率，即 $err = \mathbb{E_w}[I\{Y=g_m(x)\}| x]$ ，因此有

$$
\begin{aligned}
    & \mathbb{E}[L(Y, F(x) + \eta g_m(x))| x]  \\
    &= e^{-\eta}\mathbb{E_w}[I\{Y=g_m(x)\}| x] + e^{\eta}\mathbb{E_w}[I\{Y \neq g_m(x)\} | x] \\ 
    &=  e^{-\eta} (1-err) + e^{\eta} err \\
    & \ge 2 \sqrt{(err(1-err)}  \quad \text{(均值不等式)}
\end{aligned}
$$ 

当且仅当 $e^{-\eta} (1-err) =  e^{\eta} err$ 即 

$$
\eta = 2 \log \frac{1-err}{err}
$$ 

时等号成立。

由 $g_m(x)$ 的计算和 $\eta_m$ 的更新规则可以看出，当损失函数是指数损失时，

# References

1. [Greedy function approximation: A gradient boosting machine.](http://docs.salford-systems.com/GreedyFuncApproxSS.pdf)
2. [Gradient descent - Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
3. [Gradient boosting - Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting)
4. 李航. 统计学习方法
