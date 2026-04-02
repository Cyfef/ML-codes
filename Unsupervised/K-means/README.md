# K-Means Clustering

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $n$ | $\mathbb{R}$ | Number of samples | 
| $d$ | $\mathbb{R}$ | Input dimension | 
| $K$ | $\mathbb{R}$ | Number of clusters | 
| $x_i$ | $\mathbb{R}^d$ | $i$-th data point | 
| $X$ | $\mathbb{R}^{n\times d}$ | $(x_1,\cdots,x_n)^\top$ | 
| $\mu_j$ | $\mathbb{R}^d$ | Center of cluster $j$ | 
| $c_i$ | $\mathbb{R}$ | Cluster assignment of $x_i$ ;$\{1,\dots,k\}$ | 
| $C_j$ | - | Set of points assigned to cluster $j$ | 
| $\|\cdot\|_2$ | - | Euclidean norm |

---

## Formulas

- Objective Function

  $$
  \min_{{\mu_j}, {c_i}}
  \sum_{i=1}^n \left| x_i - \mu_{c_i} \right|_2^2
  $$

---

- Equivalent Matrix Form

  Let $( r_{ij} \in \{0,1\} )$ be an indicator:

  $$
  r_{ij} = 1,  \text{if } c_i = j  ;\quad0,  \text{otherwise}
  $$

  Then the objective becomes:

  $$
  \sum_{i=1}^n \sum_{j=1}^k
  r_{ij} |x_i - \mu_j|_2^2
  $$

---

- Training Process

  1. Initialize centers
  2. Repeat until convergence:

     * Assignment step:

        Assign each point to the nearest cluster center:

        $$
        c_i =
        \argmin_{j \in {1,\dots,k}}
        \left| x_i - \mu_j \right|_2^2
        $$

     * Update step:

        Update each cluster center as the mean of its assigned points:
       $$
       \mu_j =
       \frac{1}{|C_j|} \sum_{i \in C_j} x_i
       $$

---

- Stopping Criteria

  * Centers do not change
  * Assignments do not change
  * Objective improvement below threshold
  * Maximum number of iterations reached

