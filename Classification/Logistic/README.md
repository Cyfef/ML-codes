# Logistic Regression

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $w$ | $\mathbb{R^d}$ | the weights of each feature |
| $b$ | $\mathbb{R}$ | the bias term |
| $\hat{w}$ | $\mathbb{R^{d+1}}$ | $(w^\top,b^\top)^\top$ |
| $\sigma(\cdot)$ | - | sigmoid function |
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | the number of features |   
| $x_i$ | $\mathbb{R^d}$ | the d-D features of each sample |
| $\hat{x_i}$ | $\mathbb{R^{d+1}}$ | $(x_i^\top,1)^\top$ |
| $X_0$ | $\mathbb{R^{n \times d}}$ | $(x_1,\cdots,x_n)^\top$ |
| $X$ | $\mathbb{R^{n \times (d+1)}}$ | $(X_0,1)$ |
| $y_i$ | $\mathbb{R}$ | the classification label (0/1) | 
| $y$ | $\mathbb{R^n}$ | $(y_1,\cdots,y_n)^\top$ |  

## Formulas

- prediction

  $$
    \hat{y_i} = \sigma(w^\top x_i +b) = \sigma(\hat{w}^\top \hat{x_i})
    \\
    \hat{y} = (\hat{y_1},\cdots,\hat{y_n})^\top = \sigma(X\hat{w})
  $$

- loss function 

  $$
    \mathcal L(\hat w) = -\sum_{i\in[n]}y_i\log\sigma(\hat w^\top\hat x_i)+(1-y_i)\log(1-\sigma(\hat w^\top\hat x_i))
    = -\hat{w}^\top X^\top y + \sum_{i\in[n]} \log(1+e^{\hat{w}^\top \hat{x_i}})
  $$

- optimization

  $$
    \frac{\partial\mathcal L(\hat w)}{\partial \hat w} = -X^\top y + \sum_{i\in[n]} \sigma(\hat{w}^\top \hat{x_i})\hat{x_i}
  $$