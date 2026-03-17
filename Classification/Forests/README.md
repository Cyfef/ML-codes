# Random Forests

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | the number of features of a sample |
| $x_i$ | - | a training data |
| $y_i$ | $\mathbb{R}$ | the classification label (-1/+1) of $x_i$ |
| $D$ | - | $\{x_1,\cdots,x_n\}$ |
| $M$ | $\mathbb{R}$ | the number of base learners |
| $f_m$ | - | base learner (decision tree) |
| $F$ | - | bagged ensemble predictor |

## Formulas

- Bagging process:
  
  1. Bootstrap sampling of data: 
    
     For $m\in[M]$ , random sample a bootstrap dataset $D^{(m)}$ from $D$ , train $f_m$ on $D^{(m)}$

  2. Random feature selection: 

        When splitting a node, randomly select a subset of features and choose the best split 

  3. Bagged emsemble predictor:
    
    $$
        F(x)=\mathrm{sign}(\dfrac{1}{M}\sum_{m=1}^Mf_m(x))
    $$