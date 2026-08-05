# Autoencoder (AE)

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $d$ | $\mathbb{R}$ | the dimension number of a high-dimensional data |   
| $h$ | $\mathbb{R}$ | the dimension number of the low-dimensional representation |
| $x$ | $\mathbb{R}^d$ | a high-dimensional data |
| $z$ | $\mathbb{R}^h$ | a low-dimensional representation |
| $\hat{x}$ | $\mathbb{R}^d$ | the high-dimensional data reconstructed from $z$ |
| $f_\theta(\cdot)$ | - | $f_\theta:\, \mathbb R^d \to \mathbb R^h$ |
| $g_\psi(\cdot)$ | - | $g_\psi:\, \mathbb R^h \to \mathbb R^d$ |
| $n$ | $\mathbb{R}$ | the number of training data |
| $x_i$ | $\mathbb{R}^d$ | d-D training data |

## Formulas

- Encoder
    
    $$
        z=f_\theta(x)
    $$

- Decoder

    $$
        \hat{x}=g_\psi(z)
    $$

- Optimization

    $$
        \argmin_{\theta,\psi}
    \sum_{i=1}^{n}\bigl\|x_i - \hat{x}_i\bigr\|^2,
    \hat{x}_i = g_\psi\!\bigl(f_\theta(x_i)\bigr)
    $$