# PCA

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | the number of dimensions of a data |
| $x_i$ | $\mathbb{R^d}$ | the number of samples |
| $\bar{x}$ | $\mathbb{R^d}$ | $\frac{1}{n}\sum_{i=1}^nx_i$ |    
| $X$ | $\mathbb{R^{n \times d}}$ | $(x_1,\cdots,x_n)^\top$ |
| $\hat{X}$ | $\mathbb{R^{n \times d}}$ | $(x_1-\bar{x},\cdots,x_n-\bar{x})^\top$ |
| $\Sigma$ | $\mathbb{R^{d \times d}}$ | $\frac{1}{n}\hat{X}^\top \hat{X}$ |
| $u_i$ | $\mathbb{R^d}$ | eigenvector whose norm is 1 |
| $U$ | $\mathbb{R^{d \times d}}$ | $(u_1,\cdots,u_n)$ |
| $\lambda_i$ | $\mathbb{R}$ | eigenvalues |
| $\Lambda$ | $\mathbb{R^{d \times d}}$ | $\mathrm{diag}\{\lambda_1,\cdots,\lambda_n\}$ |
| $K$ | $\mathbb{R}$ | the principal dimensions |
| $U_{1:K}$ | $\mathbb{R}^{d \times K}$ | the projection matrix |

## Formulas

- Orthogonal Eigen-Decomposition of the Covariance

  $$
    \Sigma=U^\top\Lambda U
  $$

- Dimensionality Reduction
  
  $$
    \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n  
  $$
  with $u_i$ corresponding to $\lambda_i$ ,
  $$
    U_{1:K}=(u_1,\cdots,u_K)
  $$
  the projected data matrix
  $$
    XU_{1:K}\in \mathbb{R}^{n \times K}
  $$