# Softmax Regression

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $K$ | $\mathbb{R}$ | the number of classification labels |
| $W$ | $\mathbb{R}^{K\times d}$ | the weights matrix |
| $b$ | $\mathbb{R}^K$ | the bias vector |
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | the number of features |   
| $x_i$ | $\mathbb{R^d}$ | the d-D features of each sample |
| $X$ | $\mathbb{R^{n \times d}}$ | $(x_1,\cdots,x_n)^\top$ |
| $y_i$ | $\mathbb{R}^K$ | the classification one-hot vector | 
| $Y$ | $\mathbb{R}^{n \times K}$ | $(y_1,\cdots,y_n)^\top$ |  
| $\mathrm{softmax}(\cdot)$ | - | softmax function |
| $1_n$ | $\mathbb{R}^n$ | $1_n=(1,\cdots,1)^\top$ |



## Formulas

- Predict
  
  $$
    \hat{z}=Wx+b 
  $$

  $$
    \hat{s}=\mathrm{softmax}(\hat{z})
  $$

  $$
    \hat{y}=\argmax_i \hat{s_i}
  $$

  $$
    \hat{Z} = XW^\top + \mathbf{1}_n b^\top 
  $$
  
  $$
    \hat{S} = \mathrm{softmax}(\hat{Z}) ,\text{axis}=1
  $$

  $$
    \hat{y} = \argmax(S, \text{axis}=1)
  $$
- Cross-Entropy Loss
  
  $$
    L(y, \hat{s}) = -\log (y\cdot \hat{s})
  $$

  $$L(Y, \hat{S}) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{K} Y_{ij} \log(\hat{S_{ij}})$$

- Optimization

    $$
        \dfrac{\partial L}{\partial W}=(\hat{s}-y)x^\top
    $$

    $$
        \dfrac{\partial L}{\partial b}=\hat{s}-y
    $$

    $$\frac{\partial L}{\partial W} = \frac{1}{n} (S - Y)^\top X$$

    $$\frac{\partial L}{\partial b} = \frac{1}{n} (S - Y)^\top \mathbf{1}_n$$