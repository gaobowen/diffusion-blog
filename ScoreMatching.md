

score-based generative model的核心idea是:
```
通过在噪声扰动后的大规模数据集(noise-perturbed data distributions)上学习一种score functions (gradients of log probability density functions)(得分函数, 一种对梯度的对数似然估计)，用朗之万进行采样得到符合训练集的样本.
```
![](./ScoreMatching/smld.jpg)


对 log PDF 的梯度进行建模得到一个名为 score function 的量.



扩散模型与能量模型，Score-Matching和SDE，ODE的关系  
https://zhuanlan.zhihu.com/p/576779879


Score视角下的生成模型  
http://myhz0606.com/article/ncsn