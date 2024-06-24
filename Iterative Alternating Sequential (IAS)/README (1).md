# complex-signal-ias
A Python implementation of an iterative alternating sequential (IAS) algorithm for the reconstruction of a complex-valued signal with sparsity promotion.

## Basic problem

This code is designed for solving optimization problems of the form
```math
\text{argmin}_{(r, \phi)} E(r, \phi) \coloneqq \text{argmin}_{(r, \phi)} \left\{ \frac{1}{\sigma^2} \| F D_{ e^{i \phi} } r - y \|_2^2 + \frac{\lambda}{2} r^T R^T R r    \right\}.
```
Here $F \in \mathbb{R}^{n \times n}$ is the discrete Fourier transform, $r \in \mathbb{R}^{n}\_{\geq 0}$, $\phi \in [-\pi, \pi)^n$, and $D_{e^{i \phi}} \coloneqq \text{diag}(e^{i \phi})$ where the exponential is taken element-wise. This optimization corresponds to the MAP estimate of an associated Bayesian model. $R \in \mathbb{R}^{k \times n}$ may be chosen to enforce sparsity in a linear transformation of the magnitude of the signal, e.g., $R$ could be the identity or a discrete gradient operator. The parameter $\lambda$ is a regularization parameter that may be tuned.

We can solve this problem via a coordinate descent method. We observe that for any fixed $r$ the minimizer of $E(r, \phi)$ is achieved by
```math
\phi^\star = \text{angle}(F^H y).
```
Then, fixing $\phi = \phi^\star$, the optimal $r$ is given by
```math
r^\star = \text{argmin}_{r \geq 0} \left\{ \frac{1}{\sigma^2} \| F D_{ e^{i \phi^\star} } r - y \|_2^2 + \frac{\lambda}{2} r^T R^T R r    \right\}
```
which we solve using the gradient projected conjugate gradient (GPCG) method presented in [[1]](#1). This code relies on the implementation of GPCG [here](https://github.com/jlindbloom/gradient-projected-conjugate-gradient).

## Enforcing sparsity

To enforce sparsity under the linear transformation of the magnitude, we modify the _iterative alternating sequential_ (IAS) algorithm of [[2]](#2). Here we solve the problem
```math
\text{argmin}_{(r, \phi, \beta)} E(r, \phi, \beta) \coloneqq \text{argmin}_{(r, \phi, \beta)} \left\{ \frac{1}{\sigma^2} \| F D_{ e^{i \phi} } r - y \|_2^2 + \frac{1}{2} r^T R^T D_{\beta}^{-1} R r  - \log \pi(\beta) \right\}
```
where $\beta \in \mathbb{R}^{k}\_{\geq 0}$ are learned local variance parameters, and $\pi(\beta)$ is a generalized Gamma hyper-prior density given by
```math
\pi(\beta) \propto \prod_{i=1}^k \beta_i^{r s - 1} \exp\left\{ - \left( \frac{\beta_i}{\vartheta} \right)^r \right\}
```
for real parameters $r \neq 0$, s > 0$, $\vartheta > 0$. This code only supports the choices of $r = -1$ or $r = 1$. 

The minimizer corresponds to the MAP estimate of an associated Bayesian hierarchical model. As in the previous case, the optimal $\phi$ can be found by computing $\phi^\star = \text{angle}(F^H y)$. Then, to find the optimal $r$ and $\phi$, we apply coordinate descent until convergence (details omitted). Note that for $r = -1$, the problem is nonconvex and we may obtain only a local minima.





## References
<a id="1">[1]</a> 
Moré, J., & Toraldo, G. (1991). On the Solution of Large Quadratic Programming Problems with Bound Constraints. SIAM Journal on Optimization, 1(1), 93-113. 

<a id="2">[2]</a> 
Daniela Calvetti, Monica Pragliola, Erkki Somersalo, & Alexander Strang (2020). Sparse reconstructions from few noisy data: analysis of hierarchical Bayesian models with generalized gamma hyperpriors. Inverse Problems, 36(2), 025010.







