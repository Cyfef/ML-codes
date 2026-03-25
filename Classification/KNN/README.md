# K-Nearest Neighbors (KNN)

## Settings

| symbol | shape | meaning |
| :--- | :--- | :--- |
| $n$ | $\mathbb{R}$ | the number of training samples |
| $d$ | $\mathbb{R}$ | input dimension |
| $K$ | $\mathbb{R}$ | number of nearest neighbors (hyperparameter) |
| $x^{(i)}$ | $\mathbb{R}^d$ | the $i$-th training input |
| $y^{(i)}$ | $\mathbb{R}$ | the $i$-th training label/target |
| $x_*$ | $\mathbb{R}^d$ | a test input |
| $\mathcal{N}_K(x_*)$ | - | the set of $k$ nearest neighbors of $x_*$ |
| $\hat{y}_*$ | $\mathbb{R}$ | predicted output |
| $d(x,x')$ | - | distance function |


## Formulas

- Distance Function

    1. Euclidean distance:

        $$
            d(x,x') = \|x - x'\|_2
        $$

    2. Manhattan distance:

        $$
            d(x,x') = \|x - x'\|_1
        $$

- Neighbor Selection

    For a test point $x_*$, compute distances to all training points:

    $$
        d_i = d(x_*, x^{(i)}), \quad i = 1, \dots, n
    $$

    Select the indices of the $K$ smallest distances:

    $$
        \mathcal{N}_K(x_*) = \operatorname{arg\,topK}_{i \in \{1,\dots,n\}} \; \text{smallest } d_i
    $$

- KNN for Classification

    Predict the class by majority vote:

    $$
        \hat{y}_* = \operatorname{mode}\left( \{ y^{(i)} \mid i \in \mathcal{N}_K(x_*) \} \right)
    $$

- Prediction Process

    For each test input $x_*$:

    1. Compute distances $d(x_*, x^{(i)})$ for all $i$
    2. Find $\mathcal{N}_K(x_*)$
    3. Classification: majority vote

