# SVM

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $w$ | $\mathbb{R^d}$ | the weights of each feature |
| $b$ | $\mathbb{R}$ | the bias term |
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | the number of features |   
| $x_i$ | $\mathbb{R^d}$ | the d-D features of each sample |
| $y_i$ | $\mathbb{R}$ | the classification label (-1/1) | 
| $\alpha_i$ | $\mathbb{R}$ | Lagrange multiplier (Dual variables) |
| $\alpha^*$ | $\mathbb{R}^n$ | the optimization of the dual problem |
| $\alpha_i^*$ | $\mathbb{R}$ | $\alpha^*=(\alpha_1^*,\cdots,\alpha_n^*)$
| $w^*$ | $\mathbb{R^d}$ | the optimal solution of $w$ |
| $b^*$ | $\mathbb{R}$ | the optimal solution of $b$ |
| $d'$ | $\mathbb{R}$ | the high dimension which is used for kernel method
| $\varphi(x)$ | - | a function mapping: $\mathbb{R^d}\rightarrow \mathbb{R}^{d'}$
| $K(x_i,x_j)$ | - | kernel function |
| $\xi_i$ | $\mathbb{R}$ | slack variables |


## Formulas

- Kernel functions
  $$
  K(x_i,x_j)=\varphi(x_i)^\top \varphi(x_j)
  $$
  1. Linear kernel:
    $$
        K(x,z)=x^\top z
    $$
  2. Polynomial kernel:
    $$
        K(x,z)=(x^\top z+1)^p
    $$ 
  3. Gaussian/RBF kernel:
    $$
        K(x,z)=e^{-\dfrac{\|x-z\|^2}{2\sigma^2}}
    $$

- Primal form of SVM optimization:
  
    $$
        \argmax_{w,b} \frac{1}{\|w\|}, \quad \text{s.t. } y_i (w^\top x_i + b) \geq 1 ,\forall i
    $$

- Dual problem of SVM optimization:
  
  $$
    \argmax_{\alpha \ge 0}\quad W(\alpha) = \sum_{i \in [n]} \alpha_i - \frac{1}{2} \sum_{i,j \in [n]} \alpha_i \alpha_j y_i y_j K(x_i,x_j),\\
        \text{s.t.}\quad
         \sum_{i \in [n]} \alpha_i y_i = 0
  $$

- Sequential Minimal Optimization (SMO)
  
  1. Randomly select a pair $(\alpha_i,\alpha_j)$ , keeping $\alpha_k (k\neq i,j)$ fixed
  2. Equality constraint becomes:

    $$
        \alpha_i y_i + \alpha_j y_j = -\sum_{k \neq i,j} \alpha_k y_k = \text{constant}
    $$
  3. Original optimization problem reduces to a 1-D quadratic programming problem in $\alpha_i$ , which is convex
  4. Repeating this process iteratively over different pairs, the algorithm converges to the optimal solution of the dual problem

- Optimal solution
  
  $$
    w^* = \sum_{i \in [n]} \alpha_i^* y_i x_i,
    \qquad
    b^* = y_i - w^{*\top}x_i
    \quad \text{for any support vector } \alpha_i^* > 0
  $$

- Prediction

  $$
    f(x)=w^{*\top} x+b^*=\sum_{i \in \mathrm{S.V.}} \alpha_i^* y_i \, K(x_i,x) + b^* > 0
  $$ 

- Soft-Margin SVM
  
  $$
    \argmin_{w,b}\ \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \max(0,\, 1 - y_i(w^\top x_i + b))
  $$