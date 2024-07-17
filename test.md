思考：总体与部分的定义，需要外部指定。【自指】
群论理解模型的性质，不用映射思想。


模型去噪解码的最终目的是为了从噪声中推理所有此类未知图片，$X$ 表示所有此类型的未知图片，$x_0$ 表示抽样样本图片。  
生成模型的动机是从已有的数据 生成 更多的未知数据，即预测分布 $P(X|x_0)$，同时我们还想知道已知数据 $x_0$ 的分布。  
这里我们假设所有图片的分布都可以用高斯分布来表示，$P(X|x_0)$可以看成一个复杂的混合高斯模型，根据“平均场理论”我们可以用一系列分布累乘的形式近似表示此分布 $Q(X)=\prod_i Q_i(X_i) \approx P(X|x_0)$。用KL散度来测量 $Q(X)$与$P(X|x_0)$ 的相似程度有： 
$$KL(Q(X)||P(X|x_0))=\int Q(X) log\dfrac{Q(X)}{P(X|x_0)} dX = - \int Q(X) log\dfrac{P(X|x_0)}{Q(X)} dX \\
= - \int Q(X) log\dfrac{P(X|x_0)P(x_0)}{Q(X)P(x_0)} dX \\
= - \int Q(X) \big[log{P(X,x_0)}-log{Q(X)}-logP(x_0) \big] dX \\
= - \int Q(X) \big[log{P(X,x_0)}-log{Q(X)}\big]dX + logP(x_0) \\
$$

$$
\begin{aligned}
KL(Q(X)||P(X|x_0)) &= - (\mathbb{E}_{Q(x)}[log{P(X,x_0)}]- \mathbb{E}_{Q(x)}[log{Q(X)}]) + logP(x_0) \\

KL(Q(X)||P(X|x_0)) &= - (H[Q(X)]- H[Q(X),P(X,x_0)]) + logP(x_0) \\

KL(Q(X)||P(X|x_0)) &= KL(Q(X)||P(X,x_0)) + logP(x_0)
\end{aligned}
$$

令$ELBO(Q)=H[Q(X)]- H[Q(X),P(X,x_0)]$，$H[Q],H[Q,P]$ 表示熵和交叉熵，$logP(x_0)$可以看作是定值，为了最小化 $KL(Q(X)||P(X|x_0))$，需要最大化 $ELBO$ ，则有 $logP(x_0) \geq ELBO(Q)$，此时 $ELBO(Q)$ 称为 $logP(x_0)$ 的证据下界。  

$$KL(Q(X)||P(X,x_0)) = \int Q(X)log \dfrac{Q(X)}{P(X,x_0)} dX \\
= \int Q(X)log \dfrac{Q(X)}{P(X|x_0)P(x_0)} dX \\
= \int Q(X)log \dfrac{Q(X)}{P(X|x_0)} dX - logP(x_0) \\
KL(Q(X)||P(X,x_0)) = KL(Q(X)||P(X|x_0)) + [-logP(x_0)] \\

KL(Q(X)||P(X,x_0))P(x_0) = KL(Q(X)||P(X|x_0))P(x_0) + [-P(x_0)logP(x_0)] \\

KL(Q(X)||P(X,x_0))P(x_0) = KL(Q(X)||P(X|x_0))P(x_0) + H[P(x_0)] \\
H[P(x_0)] > 0
\quad \\
对于关于X函数，- logP(x_0) 看作常数值的情况下\\
min\{KL(Q(X)||P(X,x_0))\} 和 min\{KL(Q(X)||P(X|x_0))\} 等价，\\
且 KL(Q(X)||P(X|x_0)) \geq logP(x_0) \\
具体使用哪个看情况 \\

KL(Q(X)||P(X|x_0))=\int Q(X) log \dfrac{Q(X)}{P(X|x_0)}


\quad \\
\quad \\
\quad \\
logP(x_0) = KL(Q(X)||P(X|x_0)) - KL(Q(X)||P(X,x_0)) \\
logP(x_0) \leq KL(Q(X)||P(X|x_0)) 
$$