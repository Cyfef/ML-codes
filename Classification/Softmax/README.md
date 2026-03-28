# Softmax Regression

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $K$ | $\mathbb{R}$ | the number of classification labels |
| $W$ | $\mathbb{R}^{K\times d}$ | the weights matrix |
| $b$ | $\mathbb{R}^K$ | the bias vector |
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | the number of features |   
| $x_i$ | $\mathbb{R}^d$ | the d-D features of each sample |
| $X$ | $\mathbb{R^{n \times d}}$ | $(x_1,\cdots,x_n)^\top$ |
| $y_i$ | $\mathbb{R}^K$ | the classification one-hot vector | 
| $Y$ | $\mathbb{R}^{n \times K}$ | $(y_1,\cdots,y_n)^\top$ |  
| $\mathrm{softmax}(\cdot)$ | - | softmax function |
| $\mathbf{1}_n$ | $\mathbb{R}^n$ | $(1,\cdots,1)^\top$ |



## Formulas

- Predict

  $$
    \hat{Z} = XW^\top + \mathbf{1}_n b^\top 
  $$
  
  $$
    \hat{S} = \mathrm{softmax}(\hat{Z}) ,\text{axis}=1
  $$

  $$
    \hat{y} = \argmax(\hat{S}, \text{axis}=1)
  $$
- Cross-Entropy Loss

  $$L(Y, \hat{S}) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{K} Y_{ij} \log(\hat{S_{ij}})$$

- Optimization

    $$\frac{\partial L}{\partial W} = \frac{1}{n} (\hat{S} - Y)^\top X$$

    $$\frac{\partial L}{\partial b} = \frac{1}{n} (\hat{S} - Y)^\top \mathbf{1}_n$$