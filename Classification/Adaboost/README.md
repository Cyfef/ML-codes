# Adaboost

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | the number of features of a sample |
| $x_i$ | - | a training data |
| $y_i$ | $\mathbb{R}$ | the classification label (-1/+1) of $x_i$ |
| $D$ | - | $\{x_1,\cdots,x_n\}$ |
| $T$ | $\mathbb{R}$ | the iterations of training |
| $t$ | $\mathbb{R}$ | the iter number $(1,\cdots,T)$ |
| $w_i^{(t)}$ | $\mathbb{R}$ | sample weight of $x_i$ in iter $t$ |
| $f_t(x)$ | - | the weak classifier in iter $t$ |
| $\alpha_t$ | $\mathbb{R}$ | the weight of weak classifier $f_t(x)$ |
| $F_t(x)$ | - | the ensemble model in iter $t$ |



## Formulas

- Adaboost process
  
  1. Initialize sample weights and ensemble model :
    
    $$
        w_i^{(1)}=\frac{1}{n}, \forall i \in \{1,\cdots,n\}
    $$

    $$
        F_0(x)=0
    $$

  2. In iter $t\in \{1,\cdots,T\}$ :
    
        a. Use $D$ and $\{w_i^{(t)}\}_{i=1}^n$ to train $f_t(x):\mathbb{R}^d \rightarrow \{+1,-1\}$

        b. Calculate $f_t(x)$ 's weighted classification error on $D$ :

        $$
            \epsilon_t = \sum_{i\in [n]} w_i^{(t)}1(f_t(x_i) \neq y_i)
        $$ 

        if $\epsilon_t>0.5$ , break 

        c. Compute a weight $\alpha_t$ for $f_t(x)$ :

        $$
        \alpha_t=\frac{1}{2}\log\frac{1-\epsilon_t}{\epsilon_t}
        $$

        d. Update sample weights :

        $$
        \bar{w_i}^{(t+1)}=\bar{w_i}^{(t)} \cdot e^{-\alpha_t y_i f_t(x_i)}
        $$

        $$
        w_i^{(t+1)}=\frac{\bar{w_i}^{(t+1)}}{\sum_{j\in [n]}\bar{w_j}^{(t+1)}}
        $$


  3. Combine $T$ classifiers :

    $$
        F_T(x)=\sum_{t\in [T]}\alpha_t f_t(x)
    $$