[toc]

安装：

```
pip install lightgbm
```

```
def __init__(self, boosting_type='gbdt', num_leaves=31, max_depth=-1,
             learning_rate=0.1, n_estimators=100,
             subsample_for_bin=200000, objective=None, class_weight=None,
             min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
             subsample=1., subsample_freq=0, colsample_bytree=1.,
             reg_alpha=0., reg_lambda=0., random_state=None,
             n_jobs=-1, silent=True, importance_type='split', **kwargs):
```


1. boosting_type="gbdt"# 提升树的类型 gbdt,dart,goss,rf
2. num_leaves=31 #树的最大叶子数，对比xgboost一般为2^(max_depth)
3. max_depth=-1#最大树的深度
4. learning_rate#学习率
5. n_estimators=10: 拟合的树的棵树，相当于训练轮数
6. subsample=1.0: 训练样本采样率 行
7. colsample_bytree=1.0: 训练特征采样率 列
8. subsample_freq=1: 子样本频率
9. reg_alpha=0.0: L1正则化系数
10. reg_lambda=0.0: L2正则化系数
11. random_state=None: 随机种子数
12. n_jobs=-1: 并行运行多线程核心数
13. silent=True: 训练过程是否打印日志信息
14. min_split_gain=0.0: 最小分割增益
15. min_child_weight=0.001: 分支结点的最小权重

# References
1. [LGBMClassifier参数_Python_starmoth的博客-CSDN博客](https://blog.csdn.net/starmoth/article/details/84586709)
2. 官方文档: [lightgbm.LGBMClassifier — LightGBM 2.3.2 documentation](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.fit)

