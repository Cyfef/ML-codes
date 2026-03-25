# Support Vector Machine (SVM)

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | the number of features |   
| $d'$ | $\mathbb{R}$ | feature dimension after mapping |
| $x_i$ | $\mathbb{R^d}$ | the d-D features of each sample |
| $y_i$ | $\mathbb{R}$ | the classification label; (-1/1) | 
| $w$ | $\mathbb{R^{d'}}$ | weight vector in feature space |
| $b$ | $\mathbb{R}$ | the bias term |
| $\varphi(x)$ | - | feature mapping: $\mathbb{R^d}\rightarrow \mathbb{R}^{d'}$ |
| $K(x_i,x_j)$ | - | kernel function |
| $\alpha_i$ | $\mathbb{R}$ | Dual variables |
| $\alpha^*$ | $\mathbb{R}^n$ | optimal dual solution |
| $\alpha_i^*$ | $\mathbb{R}$ | the i-th component of $\alpha^*$ |
| $w^*$ | $\mathbb{R^{d'}}$ | optimal weight |
| $b^*$ | $\mathbb{R}$ | optimal bias |
| $\xi_i$ | $\mathbb{R}$ | slack variable |
| $C$ | $\mathbb{R}$ | regularization parameter |



## Formulas

### 1.Kernel Trick

The SVM depends only on inner products. Using a feature map 
$\varphi: \mathbb{R}^d \rightarrow \mathbb{R}^{d'}$, we define:

$$
K(x_i,x_j)=\langle \varphi(x_i), \varphi(x_j) \rangle=\varphi(x_i)^\top \varphi(x_j)
$$

This allows us to work in high-dimensional space without explicitly computing $\varphi(x)$.

- Kernel functions
  1. Linear kernel:
    $$
        K(x,z)=x^\top z
    $$
  2. Polynomial kernel:
    $$
        K(x,z)=(x^\top z+c)^p
    $$ 
  3. Gaussian/RBF kernel:
    $$
        K(x,z)=e^{-\dfrac{\|x-z\|^2}{2\sigma^2}}
    $$
---

### 2.Hard-Margin SVM (Kernelized)

#### Primal Form

$$
\argmin_{w,b} \frac{1}{2}\|w\|^2
\quad \text{s.t. } \quad y_i (w^\top \varphi(x_i) + b) \geq 1,\ \forall i
$$

#### Dual Problem

$$
\argmax_{\alpha}\quad 
W(\alpha) = \sum_{i=1}^n \alpha_i 
- \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(x_i,x_j)
$$

$$
\text{s.t.}\quad 
\sum_{i=1}^n \alpha_i y_i = 0,\quad 
\alpha_i \ge 0
$$

---

### 3.Soft-Margin SVM (Kernelized)

#### Primal Form

$$
\argmin_{w,b,\xi} \frac{1}{2}\|w\|^2 +C \sum_{i=1}^n \xi_i
\quad \text{s.t. } \quad y_i (w^\top \varphi(x_i) + b) \geq 1-\xi_i,\quad\xi_i\ge 0,\ \forall i
$$

#### Dual Problem

$$
\argmax_{\alpha}\quad 
W(\alpha) = \sum_{i=1}^n \alpha_i 
- \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(x_i,x_j)
$$

$$
\text{s.t.}\quad 
\sum_{i=1}^n \alpha_i y_i = 0,\quad 
0 \le \alpha_i \le C
$$

---

### 4.Optimal Solution (Kernel Form)

The model is represented implicitly:

$$
f(x)=\sum_{i=1}^n \alpha_i^* y_i K(x_i,x) + b^*
$$

Only samples with $\alpha_i^* > 0$ are support vectors.

---

### 5.Bias Term

$$
b^* = y_i - \sum_{j=1}^n \alpha_j^* y_j K(x_j, x_i),
\quad \text{for any } i \text{ such that } 0 < \alpha_i^* < C
$$

---

### 6.Prediction

$$
\hat{y} = \mathrm{sign}(f(x))
$$

---

### 7.Sequential Minimal Optimization (SMO)

1. Randomly select a pair $(\alpha_i,\alpha_j)$ while fixing others  
2. Use the constraint:
   $$
      \alpha_i y_i + \alpha_j y_j = -\sum_{k \neq i,j} \alpha_k y_k = \text{constant}
   $$
3. The problem reduces to a 1-D quadratic optimization in $\alpha_i$ , which is convex 
4. Repeating this process iteratively over different pairs, the algorithm converges to the optimal solution of the dual problem