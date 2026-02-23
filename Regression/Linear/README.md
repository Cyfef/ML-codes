# Linear Regression

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $n$ | - | the number of samples |
| $d$ | - | the number of features |   
| $X_0$ | $\mathbb{R^{n \times d}}$ | the input training data  |  
| $y$ | $\mathbb{R^n}$ | the gt label |  
| $w$ | $\mathbb{R^d}$ | the weights of each feature |
| $b$ | $\mathbb{R^n}$ | the bias term |
| $X$ | $\mathbb{R^{n \times (d+1)}}$ | $(X_0,1)$ |
| $\hat{w}$ | $\mathbb{R^{d+1}}$ | $(w^\top,b^\top)^\top$ |

- prediction

  $$
    \hat{y}=X_0w+b=X\hat{w}
  $$

- loss function 

  $$
    \mathcal{L}(\hat{w})=\dfrac{1}{n}(y-\hat{y})^\top(y-\hat{y})
  $$

- closed-form solution

  $$
    \hat{w}=(X^\top X)^{-1}X^\top y
  $$