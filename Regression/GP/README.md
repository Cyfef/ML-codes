# Gaussian Process (GP)

## Settings

| symbol | shape | meaning |
| :--- | :--- | :--- |
| $n$ | $\mathbb{R}$ | Number of samples |
| $d$ | $\mathbb{R}$ | Input dimension |
| $x_i$ | $\mathbb{R}^d$ | The $i$-th input data point |
| $y_i$ | $\mathbb{R}$ | The $i$-th target value |
| $X$ | $\mathbb{R}^{n\times d}$ | $(x_1^\top,\cdots,x_n^\top)^\top$ |
| $y$ | $\mathbb{R}^n$ | $(y_1,\cdots,y_n)^\top$ |
| $f(x)$ | - | latent function |
| $m(x)$ | - | mean function; $m(x)=\mathbb{E}[f(x)]$ |
| $k(x,x')$ | - | kernel (covariance) function; $k(x,x')=\mathrm{Cov}(f(x),f(x'))$|
| $\phi(\cdot;\mu,\Sigma)$ | - | the $\mathrm{p.d.f.}$ of $N(\mu,\Sigma)$ |
| $K$ | $\mathbb{R}^{n\times n}$ | kernel matrix |
| $\mathbf{m}$ | $\mathbb{R}^n$ | Means vector |
| $\sigma^2$ | - | noise variance (hyperparameter) |
| $\ell$ | $\mathbb{R}$ | Length scale in kernel function |
| $\sigma_f^2$ | $\mathbb{R}$ | Signal variance in kernel function |
| $c$ | $\mathbb{R}$ | Bias term in polynomial kernel |
| $p$ | $\mathbb{N}$ | Degree of polynomial kernel |
| $I$ | $\mathbb{R}^{n\times n}$ | Identity matrix |
| $x_*$ | $\mathbb{R}^d$ | A test point |
| $y_*$ | $\mathbb{R}$ | Predicted value of test point |

## Formulas

- Kernel functions

    1. Linear Kernel:

        $$
            k(x,x') = x^\top x'
        $$

    2. Polynomial Kernel:

        $$
            k(x,x') = (x^\top x' + c)^p
        $$
    
    3. Squared Exponential (RBF):

        $$
            k(x,x') = \sigma_f^2 \exp\left(-\frac{1}{2\ell^2} \|x - x'\|^2\right)
        $$

- Gaussian Process Prior

    A Gaussian Process is a distribution over functions:

    $$
        f(x) \sim \mathcal{GP}\bigl(m(x), k(x,x')\bigr)
    $$

    which means that for any finite set $\{x_1, \dots, x_n\}$:

    $$
        \begin{pmatrix}
        f(x_1) \\
        \vdots \\
        f(x_n)
        \end{pmatrix}
        \sim N\bigl(\mathbf{m}, K\bigr)
    $$

    where

    $$
        \mathbf{m}_i = m(x_i), \quad K_{ij} = k(x_i, x_j)
    $$

- Likelihood (Gaussian Noise Model)

    Assume noisy observations:

    $$
        y_i = f(x_i) + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)
    $$

    Then:

    $$
        y \sim N\bigl(\mathbf{m}, K + \sigma^2 I\bigr)
    $$

- Marginal Likelihood

    $$
        p(y \mid X)
        =
        \phi\bigl(y; \mathbf{m}, K + \sigma^2 I\bigr)
    $$

- Log Marginal Likelihood (for training)

    $$
        \log p(y \mid X)
        =
        -\frac{1}{2}(y-\mathbf{m})^\top (K + \sigma^2 I)^{-1}(y-\mathbf{m})
        -\frac{1}{2}\log \det (K + \sigma^2 I)
        -\frac{n}{2}\log (2\pi)
    $$

- Prediction (Posterior)

    $$
        k_* =
        \begin{pmatrix}
        k(x_1, x_*) \\
        \vdots \\
        k(x_n, x_*)
        \end{pmatrix}, \quad
        k_{**} = k(x_*, x_*)
    $$

    The predictive distribution is:

    $$
        (f(x_*) \mid X, y, x_*)
        \sim N(\mu_*, \sigma_*^2)
    $$

    where

    $$
        \mu_* = m(x_*) + k_*^\top (K + \sigma^2 I)^{-1} (y-\mathbf{m})
    $$

    $$
        \sigma_*^2 = k_{**} - k_*^\top (K + \sigma^2 I)^{-1} k_*
    $$

- Training Process

    1. Choose a kernel function $k(x,x')$ with hyperparameters

    2. Compute kernel matrix:

        $$
            K_{ij} = k(x_i, x_j)
        $$

    3. Optimize parameters by maximizing log marginal likelihood:

        $$
            \argmax_{\theta} \log p(y \mid X; \theta)
        $$

