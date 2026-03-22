# k-Nearest Neighbors (kNN)

## Settings

| symbol | shape | meaning |
| :--- | :--- | :--- |
| $n$ | $\mathbb{R}$ | the number of training samples |
| $d$ | $\mathbb{R}$ | input dimension |
| $k$ | $\mathbb{R}$ | number of nearest neighbors |
| $x^{(i)}$ | $\mathbb{R}^d$ | the $i$-th training input |
| $y^{(i)}$ | - | the $i$-th training label/target |
| $\mathbf{X}$ | $\mathbb{R}^{n\times d}$ | training inputs |
| $\mathbf{y}$ | - | training labels/targets |
| $x_*$ | $\mathbb{R}^d$ | a test input |
| $\mathcal{N}_k(x_*)$ | - | the set of $k$ nearest neighbors of $x_*$ |
| $d(x,x')$ | $\mathbb{R}$ | distance function |
| $\hat{y}_*$ | - | predicted output |

---

## Formulas

- Distance Function

    Common choices:

    1. Euclidean distance:

        $$
            d(x,x') = \|x - x'\|_2
        $$

    2. Manhattan distance:

        $$
            d(x,x') = \|x - x'\|_1
        $$

---

- Neighbor Selection

    For a test point $x_*$, compute distances to all training points:

    $$
        d_i = d(x_*, x^{(i)}), \quad i = 1, \dots, n
    $$

    Select the indices of the $k$ smallest distances:

    $$
        \mathcal{N}_k(x_*) = \operatorname{arg\,topk}_{i \in \{1,\dots,n\}} \; \text{smallest } d_i
    $$

---

- kNN for Classification

    Predict the class by majority vote:

    $$
        \hat{y}_* = \operatorname{mode}\left( \{ y^{(i)} \mid i \in \mathcal{N}_k(x_*) \} \right)
    $$

---

- kNN for Regression

    Predict by averaging:

    $$
        \hat{y}_* = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x_*)} y^{(i)}
    $$

---

- Distance-Weighted kNN (Optional)

    Assign weights based on distance:

    $$
        w_i = \frac{1}{d(x_*, x^{(i)}) + \epsilon}
    $$

    Regression:

    $$
        \hat{y}_* =
        \frac{\sum_{i \in \mathcal{N}_k(x_*)} w_i y^{(i)}}
             {\sum_{i \in \mathcal{N}_k(x_*)} w_i}
    $$

    Classification:

    $$
        \hat{y}_* =
        \operatorname{argmax}_c
        \sum_{i \in \mathcal{N}_k(x_*),\, y^{(i)} = c} w_i
    $$

---

- Training Process

    1. Store the training dataset $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$
    2. Choose:
        - number of neighbors $k$
        - distance function $d(\cdot,\cdot)$
    3. (Optional) normalize features

---

- Prediction Process

    For each test input $x_*$:

    1. Compute distances $d(x_*, x^{(i)})$ for all $i$
    2. Find $\mathcal{N}_k(x_*)$
    3. Compute prediction:
        - classification: majority vote
        - regression: average (or weighted average)

---

- Properties

    - Non-parametric model (no explicit training)
    - Lazy learning (computation deferred to prediction)
    - Sensitive to:
        - choice of $k$
        - feature scaling
        - distance metric
    - Time complexity:
        - training: $\mathcal{O}(n)$
        - prediction: $\mathcal{O}(nd)$ per query

---

- Summary

    - kNN predicts based on **local neighborhood**
    - Simple yet effective baseline method
    - Works well for low-dimensional data
    - Performance degrades in high dimensions (curse of dimensionality)
