[toc]

# DL 中的求导技巧

## 行向量 or 列向量？

在公式推导中，样本 x 常常使用列向量表示。而在实际编程中，常常使用行向量表示，这是一个坑。刚开始不注意的时候会搞晕。

比如回归问题的公式表达为 $\hat{y} = \beta_0 + x^T \beta_1$ 

在实际编程中，会像下面这样实现

```
import numpy as np

n_samples = 10
n_features = 5

x = np.random.randn(n_samples, n_features)
beta0 = np.random.randn()
beta1 = np.random.randn(n_features, 1)

y = beta0 + np.matmul(x, beta1)
```

可以看到，虽然公式中写的是 $x^T \beta_1$ 但是在编程计算时写的是 `np.matmul(x, beta1)`。

也就是说，在公式中用 $x^T$ 来表示的，编程时统统换成 $x$，同样的，公式中的  $x$ 要在编程时换成  $x^T$ 

## 求导的技巧

### 例1： 回归

1. 先计算一个样本的情况，再拓展到一堆样本即 batch 的情况。这样做的原因有两点：
    1. DL中向量可以看做是一个最小的单位，一个向量代表这一个样本。（注意，DL中的向量通常指行向量。）。一个矩阵指是一次性将一个batch的向量输入而已。因此，先使用向量求导，之后再将向量扩展为矩阵。
2. 如何将向量计算的结果拓展为矩阵？**只需要将向量求得的结果平均就是 batch 的结果**。

下面以回归为例，演示如何将向量的问题转化为矩阵问题。

假设回归问题的模型为 

$$
\hat{y} = \beta_0 + x^T \beta_1 
$$ 

Loss 是 MSE

$$
l(y, \hat{y}) = ( y - \hat{y} )^2
$$ 

注意，上面的都是一个样本的情况。

下面来进行求导。

1. 先计算 $\frac{\partial l(y, \hat{y})}{\partial \hat{y}}$，这个就是标量求导，没有难度。
$$
\frac{\partial l(y, \hat{y})}{\partial \hat{y}} =  - 2(y - \hat{y})
$$ 


2. 再计算 $\hat{y}$  对参数的导数

$$
\begin{aligned}
    \frac{\partial \hat{y}}{\partial \beta_0}  &= 1 \\ 
    \frac{\partial \hat{y}}{\partial \beta_1} &= x
    \text{注意这是个向量}
.\end{aligned}
$$ 

根据 chain rule，有

$$
\begin{aligned}
    \frac{\partial l(y, \hat{y})}{\partial \beta_0} &= -2(y - \hat{y}) \\
    \frac{\partial l(y, \hat{y})}{\partial \beta_1} &= -2(y - \hat{y}) x
.\end{aligned}
$$ 

注意第二个式子是一个向量乘以一个标量。

上面的是一个样本的情况，下面转化成 batch 的情况。

假设有 n 个向量，则每个样本都可以像上面的那样计算导数。

$$
\frac{\partial l(y_1, \hat{y}_1)}{\partial \beta_1}, \cdots, \frac{\partial l(y_n, \hat{y}_n)}{\partial \beta_n},
$$ 

batch 版本的导数实际上就是将不同样本的导数求平均，在这种情况下可能转化成矩阵乘法的版本

$$
\begin{aligned}
    \frac{\partial l(y_{batch}, \hat{y}_{batch})}{\partial \beta_0} &= \frac{1}{n} \sum_{i=1}^{n} \frac{\partial l(y_i, \hat{y}_i)}{\partial \beta_0} \\
    &= \frac{1}{n} \sum_{i=1}^{n} -2(y_i- \hat{y}_i) \\
.\end{aligned}
$$ 

同样的，$\frac{\partial l(y_{batch}, \hat{y}_{batch})}{\partial \beta_1}$ 也可以这样计算

$$
\begin{aligned}
\frac{\partial l(y_{batch}, \hat{y}_{batch})}{\partial \beta_1} &= \frac{1}{n} \sum_{i=1}^{n} \frac{\partial l(y_i, \hat{y}_i)}{\partial \beta_1} \\
 &= \frac{1}{n} \sum_{i=1}^{n} -2(y_i - \hat{y}_i) x_i \\
 &= \frac{1}{n} [ x_1, \cdots, x_n ] [ -2 ( y_1 - \hat{y}_1 ), \cdots, ( y_n - \hat{y}_n ) ]^T \\
 &= \frac{1}{n} X \cdot (-2 (y - \hat{y})) \\
.\end{aligned}
$$ 

这里的 $\cdot$  表示矩阵乘法。可以看到，batch 情况下的样本可以转化为矩阵乘法。

和直接使用矩阵求导比起来，这里的推导略显臃肿。但是对于比较复杂的情况，这里的推导会显得比较简单。

3. 记得有的要累加。

![b363f9fd155648682450bfb2d1c23b5a.png](evernotecid://7E3AE0DC-DC71-4DDC-9CC8-0C832D6C11C2/appyinxiangcom/22483756/ENResource/p11354)

