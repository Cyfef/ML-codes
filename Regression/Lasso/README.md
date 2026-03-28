# Lasso Regression

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $n$ | - | Number of samples |
| $d$ | - | Number of features |   
| $X_0$ | $\mathbb{R}^{n \times d}$ | Input training data  |  
| $y$ | $\mathbb{R}^n$ | the gt label |  
| $w$ | $\mathbb{R}^d$ | the weights of each feature |
| $b$ | $\mathbb{R}$ | the bias term |
| $\mathbf{1}$ | $\mathbb{R}^n$ | $(1,\cdots,1)^\top$ |
| $X$ | $\mathbb{R}^{n \times (d+1)}$ | $(X_0,\mathbf{1})$ |
| $\hat{w}$ | $\mathbb{R}^{d+1}$ | $(w^\top,b)^\top$ |
| $\lambda$ | $\mathbb{R}$ | Hyperparameter for regularization term |
| $I$ | $\mathbb{R}^{(d+1) \times (d+1)}$ | Identity matrix |


## Formulas

- prediction

  $$
    \hat{y}=X\hat{w}
  $$

- loss function 

  $$
    \mathcal{L}(\hat{w})=\dfrac{1}{n}(y-\hat{y})^\top(y-\hat{y}) + \lambda \|\hat{w}\|_1
  $$

- gradient
  
  $$
    \dfrac{\partial \mathcal{L}(\hat{w})}{\partial \hat{w}} =
    -\dfrac{2}{n} X^\top (y - X\hat{w})
    + \lambda \cdot \text{sign}(\hat{w})
  $$
