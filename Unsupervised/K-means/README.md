# k-Means Clustering

## Settings

| symbol | shape | meaning |
| :--- | :--- | :--- |
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | input dimension |
| $k$ | $\mathbb{R}$ | number of clusters |
| $x^{(i)}$ | $\mathbb{R}^d$ | the $i$-th data point |
| $\mathbf{X}$ | $\mathbb{R}^{n\times d}$ | dataset |
| $\mu_j$ | $\mathbb{R}^d$ | the centroid of cluster $j$ |
| $c^{(i)}$ | $\{1,\dots,k\}$ | cluster assignment of $x^{(i)}$ |
| $C_j$ | - | the set of points assigned to cluster $j$ |
| $\|\cdot\|_2$ | - | Euclidean norm |

---

## Formulas

- Objective Function

    Minimize the within-cluster sum of squares (WCSS):

    $$
        \min_{\{\mu_j\}, \{c^{(i)}\}}
        \sum_{i=1}^n \left\| x^{(i)} - \mu_{c^{(i)}} \right\|_2^2
    $$

---

- Assignment Step (E-step)

    Assign each point to the nearest centroid:

    $$
        c^{(i)} =
        \argmin_{j \in \{1,\dots,k\}}
        \left\| x^{(i)} - \mu_j \right\|_2^2
    $$

---

- Update Step (M-step)

    Update each centroid as the mean of its assigned points:
    

    $$
        \mu_j =
        \frac{1}{|C_j|}
        \sum_{i \in C_j} x^{(i)}
    $$

---

- Equivalent Matrix Form (Optional)

    Let $r_{ij} \in \{0,1\}$ be an indicator:

    $$
        r_{ij} =
        \begin{cases}
        1, & \text{if } c^{(i)} = j \\
        0, & \text{otherwise}
        \end{cases}
    $$

    Then the objective becomes:

    $$
        \sum_{i=1}^n \sum_{j=1}^k
        r_{ij} \|x^{(i)} - \mu_j\|_2^2
    $$

---

- Convergence Property

    Each iteration (assignment + update) **does not increase** the objective:

    $$
        J^{(t+1)} \le J^{(t)}
    $$

---

- Initialization (k-means++)

    1. Choose first centroid uniformly at random
    2. For each point $x$, compute distance to nearest chosen centroid $D(x)$
    3. Sample next centroid with probability:

        $$
            p(x) \propto D(x)^2
        $$

    4. Repeat until $k$ centroids are selected

---

- Training Process

    1. Initialize centroids $\{\mu_1,\dots,\mu_k\}$
    2. Repeat until convergence:
        
        - Assignment step:

            $$
                c^{(i)} =
                \argmin_j \|x^{(i)} - \mu_j\|_2^2
            $$

        - Update step:

            $$
                \mu_j =
                \frac{1}{|C_j|} \sum_{i \in C_j} x^{(i)}
            $$

---

- Stopping Criteria

    - Centroids do not change
    - Assignments do not change
    - Objective improvement below threshold
    - Maximum number of iterations reached
