# Decision Trees

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | the number of features |
| $K$ | $\mathbb{R}$ | the number of labels |
| $x_i$ | - | a training data |
| $y_i$ | $\mathbb{R}$ | the classification label of $x_i$ $(1,\cdots,K)$ |   
| $D$ | - | $\{x_1,\cdots,x_n\}$ |
| $C_k$ | - | $\{x_i\in D:y_i=k\}$ |
| $D_{a_i}$ | - | $\{x_j\in D:A(x_j)=a_i\}$ |
| $A(x)$ | - | the value of feature $A$ which the data $x$ takes $(a_1,\cdots,a_m)$| 
| $H(D)$ | $\mathbb{R}$ | the information entropy of the dataset $D$ |
| $H(D \mid A)$ | $\mathbb{R}$ | the conditional information entropy of dataset $D$ given the feature $A$ |
| $H(D,A)$ | $\mathbb{R}$ | the information entropy of feature $A$ in dataset $D$ |


## Formulas

- entropy

  $$
    H(D)=-\sum_{k=1}^K \frac{|C_k|}{|D|} \log\frac{|C_k|}{|D|}  \\

    H(D\mid A) = \sum_{i=1}^m \frac{|D_{a_i}|}{|D|} H(D_{a_i})  \\

    H(D,A) = -\sum_{i=1}^m \frac{|D_{a_i}|}{|D|}\log\frac{|D_{a_i}|}{|D|}
  $$

- Information Gain ( Mutual Information )

  $$
    g(D,A)=H(D)-H(D\mid A)
  $$

- Information Gain Ratio

  $$
    \mathrm{GR}(D,A) = \frac{g(D,A)}{H(D,A)}
  $$

- Gini Index

  $$
    \mathrm{Gini}(D)
        = 1 - \sum_{k=1}^K \left(\frac{|C_k|}{|D|}\right)^2
        = \sum_{k=1}^K \frac{|C_k|}{|D|}\left(1 - \frac{|C_k|}{|D|}\right)
  $$

  $$
    \mathrm{Gini}(D \mid A)=\sum_{i=1}^m \frac{|D_{a_i}|}{|D|} \mathrm{Gini}(D_{a_i})
  $$