[toc]

# Naive Bayes

## 原理

### 基本原理

navie bayes 是生成模型。所谓生成模型，是以估计 $P(x,y)$ 为目标的模型。

但是 navie bayes 通常用作判别使用（生成模型有两个用途，一个是生成数据，一个是用作判别模型）

假设 $X=(x_1, \cdots, x_m)$ 是一个 m 维的随机向量，而 $y4 是一个离散随机变量。

$$
P(X, y) = P(X|y)P(y) 
$$

navie bayes 基于下面的条件独立假设：

$$
P(X|y) = \Pi_{i=1}^m P(x_i|y)
$$

由此可以得到

$$
P(X, y) = P(y) \Pi_{i=1}^m P(x_i|y)
$$

通常，我们认为 $x_i$ 也是离散分布，因此，可以使用最大似然估计来估计离散分布的分布列。

假设我们认为 $x_i \sim bernoulli(p)$，那么我们有一个参数 $p$，假设我们的样本量为 n，那么 $p$ 的估计如下

$$
\hat{p} = \frac{\sum_{j=1}^n I\{x_{ij}=1\}}{n}
$$

### 用作生成模型

虽然网上都说 Naive Bayes 是生成模型，但是没有一个人说过如何使用 Navie Bayes 生成数据，都是把 Navie Bayes 当作判别模型用，这不利于理解生成模型和判别模型的区别。

我们这里大概讲一下使用 Navie Bayes 如何生成数据。

所谓生成数据，实际上就是我们给定 y，然后从 $P(x|y)$ 这个分布中抽样来得到 $X$

由于 navie bayes 的条件独立假设，$X=(x_1, \cdots, x_m)$，当我们给定 y，比如在垃圾邮件分类问题中，我们想要生成一份垃圾邮件，就相当于我们给定 $y=1$。

我们之前已经估计出来了 $P(x_i|y)$，假设我们估计出来 $x_i|y \sim bernoulli(0.2)$，那么，我们就可以从 $bernoulli(0.2)$ 生成一个随机数并将其赋给 $x_i$。同样的，我们可以随机生成其他 $x_i$，相当于我们生成了一个样本 $X=(x_1, \cdots, x_n)$

需要注意，我们在的"生成"，是生成特征。在垃圾邮件分类任务中，我们的特征就是垃圾邮件的一堆属性，因此我们也就只能生成这些属性。

而在实际使用中，我们实际上想要生成的是垃圾邮件中的每个单词，这个用朴素贝叶斯不容易做，因为朴素贝叶斯的特征不是每个单词。

这或许就是朴素贝叶斯虽然被称为生成模型，但是却不用来生成数据，而是用作判别模型的原因。

### 用作判别模型

朴素贝叶斯更多时候是用作判别模型的。

$$
P(y|X) = \frac{\Pi_{i=1}^mP(x_i|y)P(y)}{P(X)}
$$

其中 $P(X)$ 是共有部分，不会影响概率大小的比较，因此可以去掉。我们实际上只需要对于不同的类去计算 

$$
h(X, y) = \Pi_{i=1}^mP(x_i|y)P(y)
$$

我们的预测值 $\hat{y}$ 是让 $h(X,y)$ 最大的 y，即

$$
\hat{y} = argmax_y h(X, y)
$$



## 例子

### 例：李航（CategoricalNB）

训练数据：

![2b8d60152bad4732a07a6d38aaf14484.png](evernotecid://17DACF27-DD15-47AE-A79A-0E370E882109/appyinxiangcom/22483756/ENResource/p7604)

测试数据：
x = [2, S]

#### 不使用平滑

##### CategoricalNB手推

![9153cfbb472ddd1a6fdcf76dd7f3dda0.jpeg](evernotecid://17DACF27-DD15-47AE-A79A-0E370E882109/appyinxiangcom/22483756/ENResource/p7606)

##### 代码

```python
# 准备数据
import numpy as np

X1 = [1] * 5 + [2] * 5 + [3] * 5
X2 = ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']

# 对数据进行编码成 CategoricalNB接近的样子
# X1 的 1，2，3编码为 0,1,2
X1 = [{1:0, 2:1, 3:2}.get(x) for x in X1]
# X2 的 S, M, L 编码为 0, 1, 2
X2 = [{'S':0, 'M':1, 'L':2}.get(x) for x in X2]
X_train = np.array(list(zip(X1, X2)))
y_train = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]

# [2, S] 编码为 [1, 0]
X_test = [[1,0]]

# 训练
from sklearn.naive_bayes import CategoricalNB

cnb = CategoricalNB(alpha=0)
cnb.fit(X_train, y_train)

# 输出
cnb.class_log_prior_
cnb.feature_log_prob_
cnb.predict_proba(X_test)
```

```
array([-0.91629073, -0.51082562])
[array([[-0.69314718, -1.09861229, -1.79175947],
    [-1.5040774 , -1.09861229, -0.81093022]]),
array([[-0.69314718, -1.09861229, -1.79175947],
    [-2.19722458, -0.81093022, -0.81093022]])]
array([[0.75, 0.25]])
```

和手推的结果进行比较

![3e1477519adfb96a8e4a304b4949b945.png](evernotecid://17DACF27-DD15-47AE-A79A-0E370E882109/appyinxiangcom/22483756/ENResource/p7601)


```
# cnb.class_log_prior_
np.log([6/15, 9/15])
# cnb.feature_log_prob
np.log([[3/6, 2/6, 1/6],
    [2/9, 3/9, 4/9],
    [3/6, 2/6, 1/6],
    [1/9, 4/9, 4/9]])
```

```
array([-0.91629073, -0.51082562])
array([[-0.69314718, -1.09861229, -1.79175947],
    [-1.5040774 , -1.09861229, -0.81093022],
    [-0.69314718, -1.09861229, -1.79175947],
    [-2.19722458, -0.81093022, -0.81093022]])
```

#### 使用平滑

```
from sklearn.naive_bayes import CategoricalNB

cnb = CategoricalNB()
cnb.fit(X_train, y_train)
cnb.class_log_prior_ # 这个和官网上说的不一样，官网说这个也平滑了，但是实际上没有平滑
cnb.feature_log_prob_
cnb.predict_proba(X_test) # 这个和书上计算出的不同
```

```
array([-0.91629073, -0.51082562])
[array([[-0.81093022, -1.09861229, -1.5040774 ],
        [-1.38629436, -1.09861229, -0.87546874]]),
 array([[-0.81093022, -1.09861229, -1.5040774 ],
        [-1.79175947, -0.87546874, -0.87546874]])]
array([[0.64, 0.36]])
```

这里和官网上的说明有出入，官网上说这个 class_log_prior_ 是有平滑的，但是结果说明这个并没有平滑

![3a22817ff7a33336e7f3e058590b9c2f.png](evernotecid://17DACF27-DD15-47AE-A79A-0E370E882109/appyinxiangcom/22483756/ENResource/p7603)


和手推的结果进行比较

![7171603f12f6ba06171092a7de749e50.png](evernotecid://17DACF27-DD15-47AE-A79A-0E370E882109/appyinxiangcom/22483756/ENResource/p7602)

```
# cnb.class_log_prior_ 这个没有平滑
# 如果平滑了，应该是 np.log([7/17, 10/17])
np.log([6/15, 9/15])
# cnb.feature_log_prob
np.log([[4/9, 3/9, 2/9],
    [3/12, 4/12, 5/12],
    [4/9, 1/9, 2/9],
    [2/12, 5/12, 5/12]])
```

```
array([-0.91629073, -0.51082562])
array([[-0.81093022, -1.09861229, -1.5040774 ],
       [-1.38629436, -1.09861229, -0.87546874],
       [-0.81093022, -2.19722458, -1.5040774 ],
       [-1.79175947, -0.87546874, -0.87546874]])
```


### 例：文本


数据：

![151fc812017a0732a213fa84a80137fc.png](evernotecid://17DACF27-DD15-47AE-A79A-0E370E882109/appyinxiangcom/22483756/ENResource/p7605)

#### BernoulliNB

BernoulliNB 使用的特征是 word occurence vectors

认为每一个单词出现在文档中的概率服从二项分布

##### BernoulliNB手推结果

![f63b0816e81967d96e00e61fd5f4cfe0.jpeg](evernotecid://17DACF27-DD15-47AE-A79A-0E370E882109/appyinxiangcom/22483756/ENResource/p7608)

##### BernoulliNB代码

默认使用 Laplace平滑

```
# 准备数据
import numpy as np

corpus = ['Chinese Beijing Chinese',
         'Chinese Chinese Shanghai',
         'Chinese Macao',
         'Tokyo Japan Chinese']
labels= [1, 1, 1, -1]

test_corpus = ['Chinese Chinese Chinese Tokyo Japan']

# 转化为 bag of words
from sklearn.feature_extraction.text import CountVectorizer
counter = CountVectorizer()
X_train = counter.fit_transform(corpus).toarray()
y_train = labels
X_test = counter.transform(test_corpus)

# 训练数据
from sklearn.naive_bayes import BernoulliNB


bnb = BernoulliNB()
# X_train是word count vectors，不是 word occurence vectors
# bnb 的默认参数 binarize=0 会将 word count vectors 转化为 word occurence vectors，从而满足 BernoulliNB 的假设
bnb.fit(X_train, y_train)

bnb.class_log_prior_
bnb.feature_log_prob_
bnb.predict_proba(X_test)
```

```
array([-1.38629436, -0.28768207])
array([[-1.09861229, -0.40546511, -0.40546511, -1.09861229, -1.09861229,
        -0.40546511],
       [-0.91629073, -0.22314355, -1.60943791, -0.91629073, -0.91629073,
        -1.60943791]])
array([0.80893321, 0.19106679])
```

和手推结果的对比

```
# bnb.class_log_prior
np.log([1/4, 3/4])
# bnb.feature_log_prob_
np.log([
    [1/3, 2/3, 2/3, 1/3, 1/3, 2/3],
    [2/5,4/5,1/5,2/5,2/5, 1/5]
])
p1, p2 = 16/3**6, 81/5**6
np.array([p1/(p1+p2), p2/(p1+p2)])
```

output:
```
array([-1.38629436, -0.28768207])
array([[-1.09861229, -0.40546511, -0.40546511, -1.09861229, -1.09861229,
        -0.40546511],
       [-0.91629073, -0.22314355, -1.60943791, -0.91629073, -0.91629073,
        -1.60943791]])
array([0.80893321, 0.19106679])
```

#### MultinomialNB

MultinomialNB 使用的特征是 word count vectors

##### MultinomialNB 手推

![9dd5e48224c53b31dd83a018582458d6.jpeg](evernotecid://17DACF27-DD15-47AE-A79A-0E370E882109/appyinxiangcom/22483756/ENResource/p7607)

##### multinomial代码

默认使用 Laplace平滑

```
# 准备数据
import numpy as np

corpus = ['Chinese Beijing Chinese',
         'Chinese Chinese Shanghai',
         'Chinese Macao',
         'Tokyo Japan Chinese']
labels= [1, 1, 1, -1]

test_corpus = ['Chinese Chinese Chinese Tokyo Japan']

# 转化为 bag of words
from sklearn.feature_extraction.text import CountVectorizer
counter = CountVectorizer()
X_train = counter.fit_transform(corpus).toarray()
y_train = labels
X_test = counter.transform(test_corpus)

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

mnb.class_log_prior_
mnb.feature_log_prob_
mnb.predict_proba(X_test)
```

```
array([-1.38629436, -0.28768207])
array([[-2.19722458, -1.5040774 , -1.5040774 , -2.19722458, -2.19722458,
        -1.5040774 ],
       [-1.94591015, -0.84729786, -2.63905733, -1.94591015, -1.94591015,
        -2.63905733]])
array([[0.31024139, 0.68975861]])
```

与手推结果对比

```
# mnb.class_log_prior_
np.log([1/4, 3/4])
# mnb.feature_log_prob_
np.log([
    [1/9, 2/9, 2/9, 1/9, 1/9, 2/9],
    [2/14, 6/14, 1/14, 2/14, 2/14, 1/14]]
)
p1 = (2/9)**5 * 1/4
p2 = (6/14)**3 * 1/14 * 1/14 * 3/4
# mnb.predict_proba(X_test)
np.array([p1/(p1+p2), p2/(p1+p2)])
```
