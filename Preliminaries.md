## 预备知识
#### 联合概率链式法则
$P(A_1 \cap A_2 \cap \ldots \cap A_n) = P(A_1) \times P(A_2 \mid A_1) \times \ldots P(A_n \mid A_1 \cap \ldots \cap A_{n-1})$  
$P(X_{1}X_{2}\ldots X_n) = P(X_1)P(X_2|X_1)\ldots P(X_{n}|X_{<n})$

$P(A_1 \cap A_2)$ 表示事件$A_1$和$A_2$同时发生。
$P(X_{1}X_{2})$ 表示样本$X_1$和$X_2$同时出现。

#### 贝叶斯公式
贝叶斯公式用来描述两个条件概率之间的关系，比如 P(A|B) 和 P(B|A)。按照乘法法则，$P(A \cap B)=P(A)·P(B|A)=P(B)·P(A|B)$，可以立刻导出贝叶斯定理  
$P(A|B)=\dfrac{P(B \cap A)}{P(B)}=\dfrac{P(B|A)P(A)}{P(B)}$   
“∣”读作given，即给定的意思。如 P(A∣B) 即 A given B 

#### 先验 后验 似然

- $先验=P(因)=P(\theta)$ 结果发生前, 就开始猜测(估计)原因， Prior
- $后验=P(因|果)=P(\theta|X)$ 已知结果，然后根据结果估计原因，Posterior
- $似然=P(果|因)=P(X|\theta)$ 先给定原因，根据原因来估计结果的概率分布，Likelihood
- $证据=P(果)=P(X)$ 出现结果的概率，```特别强调 这里的结果 反映的是在没有任何额外信息（即不知道结果）的情况下，出现结果的概率``` Evidence
 
这里的因果只表示事件，不表示严格的因果推断。

$$Posterior=\dfrac{Joint}{Evidence} = \dfrac{Likelihood * Prior}{Evidence}$$

更直观一点的理解：  
A：小明中午吃火锅。 B：小明晚上拉肚子。  
已知 A的出现 有利于 B的出现，则有 B的出现 也有利于 A的出现。  
$已知 P(B|A) > P(B), 则 P(A|B) > P(A)$  
$P(A|B)=\dfrac{P(B|A)}{P(B)} P(A) > P(A)$ 

#### 似然估计  
似然函数如下：
$\mathcal{L}(\theta|x)=p(x|\theta)$  
更严格地，也可写成 $\mathcal{L}(\theta|x)=p(x;\theta)$

似然性（likelihood）与概率（possibility）同样可以表示事件发生的可能性大小，但是二者有着很大的区别：

概率 
 - 是在已知参数 $\theta$ 的情况下，发生观测结果 $x$ 可能性大小；  
  
似然性   
 - 则是从观测结果 $x$ 出发，分布函数的参数为$\theta$的可能性大小；

若 已知 $x$，未知 $\theta$，对于两个参数 $\theta_1$,$\theta_2$有 $p(x|\theta_1)>p(x|\theta_2)$  

则 $\mathcal{L}(\theta_1|x)>\mathcal{L}(\theta_2|x)$

#### 最大似然估计  
最大似然估计方法（Maximum Likelihood Estimate，MLE）

最大似然估计的思想在于，对于给定的观测数据 $x$，我们希望能从所有的参数
$\theta_1$,$\theta_2$,...,$\theta_{n}$ 中找出能最大概率生成观测数据的参数 $\theta^*$作为估计结果。

 $\mathcal{L}(\theta^*|x)\geq\mathcal{L}(\theta|x),\theta=\theta_1,...,\theta_n$

$p(x|\theta^*)\geq p(x|\theta)$  

最大化概率函数的参数即可：

$\theta^*= \mathop{argmax} \limits_\theta(p|\theta)$

#### 离散型随机变量的最大似然估计  

离散型随机变量$X$的分布律为$P\{X=x\}=p(x;\theta)$，设$X_1,...,X_n$为来自$X$的样本，$x_1,...,x_n$为相应的观察值，为待估参数。在参数$\theta$下，分布函数随机取到$x_1,...,x_n$的概率为$p(x|\theta)=\prod\limits_{i=1}^n p(x_i;\theta)$, 其中$\prod$是$\pi$的大写，表示累乘。  
通过似然函数 $\mathcal{L}(\theta|x)=p(x|\theta)=\prod\limits_{i=1}^n p(x_i;\theta)$   

此时 $\mathcal{L}(\theta|x)$ 是一个关于$\theta$的函数，寻找生成$x$的最大概率， 导数等于0时，取得极值：  
$\frac{d}{d\theta} L(\theta|x) = 0$   
因为$\prod\limits_{i=1}^n p(x_i;\theta)$是累乘形式，由复合函数的单调性，对原函数取对数：  
$\frac{d}{d\theta} ln L(\theta|x) = 1/L(\theta|x) \cdot \frac{d}{d\theta} L(\theta|x)  = 0$   


#### 马尔科夫链条件概率
条件概率 $P(C,B,A)=P(C|B,A)P(B,A)=P(C|B,A)P(B|A)P(A)$  
马尔可夫链指当前状态的概率【只】与上一时刻有关，所以事件A对事件C的概率没有影响，即 $P(C|B)=P(C|B,A)$。有  
$$P(C,B,A)=P(C|B)P(B|A)P(A)$$

#### 联合分布与边缘分布
二维随机变量$\xi=(X,Y)$，$(X,Y)$的联合分布：
$$P((X,Y)=(x_i,y_j))=p_{ij}$$
$$ \sum_i \sum_j p_{ij} =1, \quad p_{ij} \geq0$$
分量$X$的概率分布称为联合分布$(X,Y)$关于$X$的边缘分布；
$$P(X=x_i)=\sum_j p_{ij}$$
分量$Y$的概率分布称为联合分布$(X,Y)$关于$Y$的边缘分布。
$$P(Y=y_j)=\sum_i p_{ij}$$
由联合分布可以推出边缘分布，但反之一般不可以，这是因为随机向量的分量之间可能有相关性。 

![](https://img-blog.csdnimg.cn/20190213135628378.png) 

`绿色为联合分布，蓝色和红色分别为 X、Y 的边缘分布`


关于其中一个特定变量的边缘分布则视为给定其他变量的条件概率分布：
$$P(x)=\sum_y P(x,y)=\sum_y P(x|y)P(y)$$ 
在边缘分布中，我们得到只关于一个变量的概率分布，而不再考虑另一变量的影响，实际上进行了降维操作。

连续型情形下，关于$X$的概率密度函数为：
$$f_X(x)=\int _{-\infin}^{+\infin} f(x,y)dy$$
Wilks 不等式  
设随机变量$(X_1,X_2,\cdots,X_n)$的联合概率分布函数为$F(x_1,x_2,\cdots,x_n)$，关于各变元的边缘分布函数是$F_i(x_i)$那么有如下不等式成立
$$F(x_1,x_2,\cdots,x_n) \leq \bigg (\prod_{i=1}^n F_i(x_i) \bigg)^{\dfrac{1}{n}}$$


https://www.math.pku.edu.cn/teachers/xirb/Courses/statprobB/psbd04.pdf


#### KL散度（Kullback-Leibler divergence） 
KL散度是两个概率分布$P$和$Q$差别的非对称性的度量。 KL散度是用来度量使用基于$Q$的分布来编码服从$P$的分布的样本所需的额外的平均比特数。典型情况下，$P$表示数据的真实分布，$Q$表示数据的理论分布。

KL散度的定义：  
$KL(P||Q)=\sum p(x) log \dfrac{p(x)}{q(x)}$

![](https://img-blog.csdn.net/20170225140215041?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYXdzMzIxNzE1MA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

若$P$为已知的真实分布，$Q$为近似分布。
(a). $KL(Q||P)$ (b). $KL(P||Q)$


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

$H(P, Q)$ 称作P和Q的交叉熵（cross entropy）, KL散度不具备对称性，也就是说 $P$对于$Q$ 的KL散度并不等于 $Q$ 对于 $P$ 的KL散度。

在信息论中，熵代表着信息量，$H(P)$ 代表着基于 $P$ 分布自身的编码长度，也就是最优的编码长度（最小字节数）。而$H(P,Q)$ 则代表着用 $P$ 的分布去近似 $Q$ 分布的信息，自然需要更多的编码长度。并且两个分布差异越大，需要的编码长度越大。所以两个值相减是大于等于0的一个值，代表冗余的编码长度，也就是两个分布差异的程度。所以KL散度在信息论中还可以称为相对熵（relative entropy）。

KL散度与交叉熵的应用
- 交叉熵通常用于监督学习任务中，如分类和回归等。在这些任务中，我们有一组输入样本和相应的标签。我们希望训练一个模型，使得模型能够将输入样本映射到正确的标签上。
- KL散度通常用于无监督学习任务中，如聚类、降维和生成模型等。在这些任务中，我们没有相应的标签信息，因此无法使用交叉熵来评估模型的性能，所以需要一种方法来衡量模型预测的分布和真实分布之间的差异，这时就可以使用KL散度来衡量模型预测的分布和真实分布之间的差异。

总结：有真实分布可以用交叉熵，没有就用KL散度。


### 高斯分布
[](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/1920px-Normal_Distribution_PDF.svg.png)

一维：$X\sim \mathcal{N}(\mu, \sigma^{2})$，其概率密度函数为：   
$$
\begin{aligned}
p(x) &= \dfrac{1}{\sqrt{2\pi\sigma^{2}}} \exp \left ({-\dfrac{1}{2}(\dfrac{x-\mu}{\sigma})^{2}} \right)  \\
&= \frac{1}{\sqrt{2\pi\sigma^2} } \exp \left[ -\frac{1}{2\sigma^2} \left( x^2 - 2\mu x + \mu^2  \right )  \right] 
\end{aligned}
$$


$\mu$ 加权平均值(期望) $E(X)=\sum_{i}{p(x_i) x_i}$  
$\sigma^2$ 方差(variance) $Var(X)=E[(X-\mu)^2]=E[X^2]-E[x]^2$  

期望方差的积分形式：  
$\mu=E(x)=\int p(x)xdx$  
$\sigma^2=E[(x-\mu)^2]=\int p(x)(x-\mu)^2dx$


期望 $E[x]$ 的另一个叫法是分布函数的 一阶矩，而 $E[x^2]$ 也叫 二阶矩 。  
由 $\sigma^2=E[(X-\mu)^2]=E[X^2]-E[x]^2$ ，有 $E[x^2]=\mu^2+\sigma^2$。

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



（3）Product: if p(x) and p(z) are Gaussian, then p(x)p(z) is proportional to a Gaussian  
（4）Marginalization: if p(x, z) is Gaussian, then p(x) is Gaussian.  
（5）Conditioning: if p(x, z) is Gaussian, then p(x | z) is Gaussian.

注：分布的等式两边不能移项

### 高斯分布的KL散度
高斯分布 $X \sim \mathcal{N}(\mu, \sigma^{2}), p(x)=\dfrac{1}{\sqrt{2\pi\sigma^{2}}}\mathrm{exp}({-\dfrac{1}{2}(\dfrac{x-\mu}{\sigma})^{2}})$
$$
KL(\mathcal{N}(\mu_1, \sigma_1^{2})||\mathcal{N}(\mu_2, \sigma_2^{2}))
= \int (2\pi \sigma_1^2)^{-1/2} \mathrm{exp}({-\dfrac{1}{2}(\dfrac{x-\mu_1}{\sigma_1})^{2}}) log \dfrac{(2\pi \sigma_1^2)^{-1/2}\mathrm{exp}({-\dfrac{1}{2}(\dfrac{x-\mu_1}{\sigma_1})^{2}})}{(2\pi \sigma_2^2)^{-1/2}\mathrm{exp}({-\dfrac{1}{2}(\dfrac{x-\mu_2}{\sigma_2})^{2}})}dx \\
\overset{\text{log拆项,log与exp抵消}}{=}\int (2\pi \sigma_1^2)^{-1/2} \mathrm{exp}({-\dfrac{1}{2}(\dfrac{x-\mu_1}{\sigma_1})^{2}})\bigg [log\dfrac{\sigma_2}{\sigma_1}-\dfrac{1}{2}(\dfrac{x-\mu_1}{\sigma_1})^{2}+\dfrac{1}{2}(\dfrac{x-\mu_2}{\sigma_2})^{2}\bigg]dx \\
\overset{记\mathcal{N}(\mu_1, \sigma_1^{2})密度函数p_1(x)}{=}\int p_1(x) \bigg [log\dfrac{\sigma_2}{\sigma_1}-\dfrac{1}{2}(\dfrac{x-\mu_1}{\sigma_1})^{2}+\dfrac{1}{2}(\dfrac{x-\mu_2}{\sigma_2})^{2}\bigg]dx
$$
分部积分：  
（1） 高斯分布积分等于1：  
$$log\dfrac{\sigma_2}{\sigma_1}\int p_1(x)dx=log\dfrac{\sigma_2}{\sigma_1}$$
（2）根据方差积分形式 $\sigma^2=\int p(x)(x-\mu)^2dx$ 有  
$$\int p_1(x) [-\dfrac{1}{2}(\dfrac{x-\mu_1}{\sigma_1})^{2}]dx = -\dfrac{1}{2\sigma_1^2} \int p_1(x) (x-\mu_1)^2dx=-\dfrac{1}{2\sigma_1^2}\sigma_1^2=-\dfrac{1}{2}$$
（3）中括号展开分别是 二阶矩、均值、常数
$$ 
\int p_1(x) [\dfrac{1}{2}(\dfrac{x-\mu_2}{\sigma_2})^{2}]dx = \dfrac{1}{2\sigma_2^2} \bigg\{ \int p_1(x)x^2dx -2\mu_2\int p_1(x)xdx + \mu_2^2 \int p_1(x)dx \bigg\} \\
= \dfrac{1}{2\sigma_2^2}(\sigma_1^2+\mu_1^2-2\mu_1\mu_2+\mu_2^2)=\dfrac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}
$$
最终结果
$$KL(\mathcal{N}(\mu_1, \sigma_1^{2})||\mathcal{N}(\mu_2, \sigma_2^{2}))=\dfrac{1}{2}\bigg[ log\dfrac{\sigma_2^2}{\sigma_1^2}-1+\dfrac{\sigma_1^2+(\mu_1-\mu_2)^2}{\sigma_2^2} \bigg]$$

特殊的
$$
KL(\mathcal{N}(\mu, \sigma^{2})||\mathcal{N}(0, 1))=\dfrac{1}{2}[-log\sigma^2-1+\mu^2+\sigma^2]
$$

#### k维高斯分布的KL散度：
$$KL(\mathcal{N}(\mu_1, \sigma_1^{2})||\mathcal{N}(\mu_2, \sigma_2^{2}))=\dfrac{1}{2}\bigg[log\dfrac{|\Sigma_2|}{|\Sigma_1|} -k +tr(\Sigma_2^{-1}\Sigma_1)+(\mu_1-\mu_2)^T\Sigma_2^{-1}(\mu_1-\mu_2) \bigg]
$$
其中 $k$ 为 维度

参考文章：  
[高斯分布的积分期望E(X)方差V(X)的理论推导](https://blog.csdn.net/chaosir1991/article/details/106864207)  
[高斯分布的KL散度](https://blog.csdn.net/hegsns/article/details/104857277)




https://blog.csdn.net/qy20115549/article/details/93074519  
https://qianyang-hfut.blog.csdn.net/article/details/86644192