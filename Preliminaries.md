## 预备知识
#### 链式法则
$P(A_1 \cap A_2 \cap \ldots \cap A_n) = P(A_1) \times P(A_2 \mid A_1) \times \ldots P(A_n \mid A_1 \cap \ldots \cap A_{n-1})$  
$P(X_{1}X_{2}\ldots X_n) = P(X_1)P(X_2|X_1)\ldots P(X_{n}|X_{<n})$

$P(A_1 \cap A_2)$ 表示事件$A_1$和$A_2$同时发生。
$P(X_{1}X_{2})$ 表示样本$X_1$和$X_2$同时出现。

#### 贝叶斯公式
贝叶斯公式用来描述两个条件概率之间的关系，比如 P(A|B) 和 P(B|A)。按照乘法法则，P(A∩B)=P(A)·P(B|A)=P(B)·P(A|B)，可以立刻导出贝叶斯定理  
$P(A|B)=\dfrac{P(B|A)P(A)}{P(B)}$   
“∣”读作given，即给定的意思。如 P(A∣B) 即 A given B 

#### 先验 后验 似然

- $先验=P(因)=P(\theta)$ 结果发生前, 就开始猜测(估计)原因， Prior
- $后验=P(因|果)=P(\theta|X)$ 已知结果，然后根据结果估计原因，Posterior
- $似然=P(果|因)=P(X|\theta)$ 先给定原因，根据原因来估计结果的概率分布，Likelihood
- $证据=P(果)=P(X)$ 出现结果的概率，```特别强调 这里的结果 反映的是在没有任何额外信息（即不知道结果）的情况下，出现结果的概率``` Evidence

$Posterior=\dfrac{Likelihood * Prior}{Evidence}$   

这里的因果只表示事件，不表示严格的因果推断。

#### KL散度（Kullback-Leibler divergence） 
KL散度是两个概率分布P和Q差别的非对称性的度量。 KL散度是用来度量使用基于Q的分布来编码服从P的分布的样本所需的额外的平均比特数。典型情况下，P表示数据的真实分布，Q表示数据的理论分布。

KL散度的定义：  
$KL(P||Q)=\sum p(x) log \dfrac{p(x)}{q(x)}$


凸函数: 直观理解，凸函数的图像形如开口向上的杯
∪，而相反，凹函数则形如开口向下的帽 ∩。
二阶导数在区间上大于等于零，就称为凸函数。例如，$y=x^2$

![](https://upload.wikimedia.org/wikipedia/commons/4/4c/%E5%87%B8%E5%87%BD%E6%95%B0%E5%AE%9A%E4%B9%89.png)

概率论中, 有延森不等式: $f(E(X)) \leq E(f(X))$
这里把 $E(X)$想象成$\dfrac{x_1+x_2}{2}$,则  
$E(f(X))=\dfrac{f(x_1)+f(x_2)}{2}$，$E(f(X))=\dfrac{f(x_1)+f(x_2)}{2}$


吉布斯不等式:  
$$
\begin{aligned}
KL(P||Q) &=\sum_x p(x) log \dfrac{p(x)}{q(x)} \\
 &= - \sum_x p(x) log \dfrac{q(x)}{p(x)} \\
 &= E[-log \dfrac{q(x)}{p(x)}] \geq  -log[E(\dfrac{q(x)}{p(x)})] \\
 &= -log [\sum_x \bcancel{p(x)}  \dfrac{q(x)} { \bcancel{p(x)} }] \\
 &= -log [\sum_x q(x)] = - log1 = 0 \\
 KL(P||Q) \geq 0
\end{aligned}
$$

概率分布的熵 (H) 的定义是：  
$H[x]=-\sum_{x} p(x)log(p(x))$  

#### KL散度与交叉熵  
$$
\begin{aligned}
KL(P||Q) &=\sum_x p(x) log \dfrac{p(x)}{q(x)} \\
&= \sum_x p(x) log(p(x)) - \sum_x p(x) log(q(x)) \\
&= - H[P] + H(P,Q)
\end{aligned}
$$

H(P, Q) 称作P和Q的交叉熵（cross entropy）, KL散度不具备对称性，也就是说 P对于Q 的KL散度并不等于 Q 对于 P 的KL散度。

在信息论中，熵代表着信息量，H(P) 代表着基于 P 分布自身的编码长度，也就是最优的编码长度（最小字节数）。而H(P,Q) 则代表着用 P 的分布去近似 Q 分布的信息，自然需要更多的编码长度。并且两个分布差异越大，需要的编码长度越大。所以两个值相减是大于等于0的一个值，代表冗余的编码长度，也就是两个分布差异的程度。所以KL散度在信息论中还可以称为相对熵（relative entropy）。

神经网络我们一般会 用交叉熵作为损失函数，在训练的过程中用 交叉熵 去逐渐靠近 真实熵。目的也是为了最小化KL散度。



### 高斯分布
[](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/1920px-Normal_Distribution_PDF.svg.png)

一维：$X\sim \mathcal{N}(\mu, \sigma^{2})$ , $p(x)=\dfrac{1}{\sqrt{2\pi\sigma^{2}}}\mathrm{exp}({-\dfrac{1}{2}(\dfrac{x-\mu}{\sigma})^{2}})$

$\mu$ 加权平均值(期望) $E(X)=\sum_{i}{p_i x_i}$  
$\sigma^2$ 方差(variance) $Var(X)=E[(X-\mu)^2]=E[X^2]-E[x]^2$  
#### 【协方差 Covariance】
用于度量两组数据的变量X和Y之间是否有线性相关性，=0不相关，>0正相关，<0负相关  
$cov(X,Y)=E[(X-E(X))(Y-E(Y))]= E[(X-\mu_X)(Y-\mu_Y)]$  
$cov(X,Y)=cov(Y,X)$  
$cov(aX,bY)=ab\;cov(Y,X)$  
#### 【协方差矩阵】  
有 n 个随机变量组成一个 n维向量 $X=\{X_1,X_2,\cdots,X_n\}$
$$
\Sigma= cov(X,X^T) :=
\left[
\begin{array}{ccc}
cov(x_1,x_1) & \cdots & cov(x_1,x_n)\\
\vdots & \ddots & \vdots\\
cov(x_1,x_n) & \cdots & cov(x_n,x_n)
\end{array}
\right]
$$


#### 【相关系数】
用于度量两组数据的变量X和Y之间的线性相关的程度。它是两个变量的协方差与其标准差的乘积之比。  
$\rho_{X,Y}=\dfrac{cov(X,Y)}{\sigma_X \sigma_Y} = \dfrac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y}$  
$cov(X,Y)=\rho\;\sigma_X \sigma_Y$

皮尔逊相关系数的变化范围为-1到1。系数的值为1意味着X和 Y可以很好的由直线方程来描述，所有的数据点都很好的落在一条直线上，且 Y 随着 X 的增加而增加。系数的值为−1意味着所有的数据点都落在直线上，且 Y 随着 X 的增加而减少。系数的值为0意味着两个变量之间没有线性关系。  

特殊的，X自己和自己的协方差 $cov(X,X)=\sigma_X^2$，相关系数 $\rho_{X,X}=1$  
若 X 和 Y 相互独立(线性不相关)，$cov(X,Y)=0$，$\rho_{X,Y}=0$


k维：$p(x)=\dfrac{1}{\sqrt{(2\pi)^{k}|\Sigma|}}\mathrm{exp}(-\dfrac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$


$\Sigma_{i,j} = cov(i,j) = \mathrm{E}[(X_i-\mu_i)(X_{j}-\mu_j)]$


#### 【封闭性】
数学中，若对某个集合的成员进行一种运算，生成的元素仍然是这个集合的成员，则该集合被称为在这个运算下闭合。

高斯分布的随机变量 线性组合 还是高斯分布:  
（1）如果$X \sim \mathcal{N}(\mu, \sigma^2)$, a 与 b 是实数，那么$aX+b \sim \mathcal{N}(a\mu+b,(a\sigma)^2)$  
（2）$X \sim \mathcal{N}(\mu_1, \sigma_1^2)$, $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$，$X,Y$是统计独立的随机变量，它们的和也服从高斯分布：  
$$X+Y \sim \mathcal{N}(\mu_1+\mu_2, \sigma_1^2 + \sigma_2^2)$$  
$$X-Y \sim \mathcal{N}(\mu_1-\mu_2, \sigma_1^2 + \sigma_2^2)$$





### 变分推断中的ELBO(证据下界)
变分推理的目标是近似潜在变量(latent variables)在观测变量（observed variables）下的条件概率。解决该问题，需要使用优化方法。在变分推断中，需要使用到的一个重要理论，是平均场理论

参考文章： 
https://blog.csdn.net/qy20115549/article/details/93074519
https://qianyang-hfut.blog.csdn.net/article/details/86644192