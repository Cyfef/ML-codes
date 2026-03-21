# Mixture of Gaussians (MoG)

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | the number of feature dimensions |   
| $K$ | $\mathbb{R}$ | the number of clusters |
| $T$ | $\mathbb{R}$ | the iterations of optimization |
| $\mu_k$ | $\mathbb{R}^d$ | the mean vector of Gaussian cluster $k$ |
| $\Sigma_k$ | $\mathbb{R^{d \times d}}$ | the covariance matrix of Gaussian cluster $k$ |
| $x$ | $\mathbb{R}^d$ | the random vector |
| $\phi(x;\mu_k,\Sigma_k)$ | - | the $\mathrm{p.d.f.}$ of $N(\mu_k,\Sigma_k)$ 
| $x_i$ | $\mathbb{R}^d$ | the training data with d-D features |
| $z_i$ | $\mathbb{R}$ | the latent variable indicating the cluster which $x_i$ belongs to ; $(1,\cdots,K)$|
| $z$ | $\mathbb{R}^n$ | $(z_1,\cdots,z_n)^\top$ |
| $\pi_k$ | $\mathbb{R}$ | the mixture coefficient |
| $\gamma_{ik}$ | $\mathbb{R}$ | the responsibility of cluster $k$ for $x_i$ |
| $\theta_k$ | - | $(\pi_k,\mu_k,\Sigma_k)$ |
| $\theta$ | - | $\{\theta_k\}_{k=1}^K$ |
| $\lambda_i$ | $\mathbb{R}$ | the final cluster that $x_i$ is assigned to |


## Formulas

- Probabilistic Model

  1. Conditional probability:
    
    $$
      (x_i \mid z_i=k) \sim N(\mu_k,\Sigma_k)
    $$
  
  2. Prior probability: 

    $$
      P(z_i=k)=\pi_k, k=1,\cdots,K
    $$
  
  3. Posterior probability:

    $$ 
      \gamma_{ik}=P(z_i=k \mid x_i)=\dfrac{\pi_k \cdot \phi(x_i;\mu_k,\Sigma_k)}{\sum_{j\in [K]}\pi_j \cdot \phi(x_i;\mu_j,\Sigma_j)}
    $$

- MLE objective:
  
  $$
    \argmax_{\theta} \sum_{i=1}^n \log p(x_i;\theta)
    =
    \argmax_{\theta} \sum_{i=1}^n \log
    \left(
        \sum_{k=1}^K \pi_k\cdot\phi(x_i;\mu_k,\Sigma_k)
    \right)
  $$

- ELBO decomposition:

  $$
        \log p(x;\theta)
        =
        \underbrace{
        \sum_{z}q(z \mid x)\cdot\left[\log \frac{p(x,z;\theta)}{q(z \mid x)}\right]
        }_{\mathcal L(q,\theta)\;\;(\text{ELBO})}
        +
        \underbrace{
        \sum_{z}q(z \mid x)\cdot\left[\log \frac{q(z \mid x)}{p(z \mid x;\theta)}\right]
        }_{\mathrm{KL}\!\left[q(z\mid x)\,\middle\|\,p(z\mid x;\theta)\right]\ge 0}
  $$

  where $q(z\mid x)$ is the variational distribution of $p(z \mid x ;\theta)$


- Optimization process
  
  1. Randomly Initialize: 
    
      $$ \theta_k^{(1)}=(\pi_k^{(1)},\mu_k^{(1)},\Sigma_k^{(1)}), \forall k\in [K] $$
  
  2. Expectation-Maximization (EM) Algorithm:

      for $t\in [T]$ ,

      - E-step (fix $\theta$): compute $\gamma_{ik}^{(t)}$ , $\forall i \in [n]$
      - M-step (fix $q$): update $\theta^{(t+1)}=(\pi_k^{(t+1)},\mu_k^{(t+1)},\Sigma_k^{(t+1)})$ , $\forall k\in[K]$

          $$
            \mu_k^{(t+1)}=\dfrac{\sum_{i\in[n]}\gamma_{ik}^{(t)}x_i}{\sum_{i\in[n]}\gamma_{ik}^{(t)}}
          $$

          $$
            \Sigma_k^{(t+1)}=\dfrac{\sum_{i\in[n]}\gamma_{ik}^{(t)}(x_i-\mu_k^{(t+1)})(x_i-\mu_k^{(t+1)})^\top}{\sum_{i\in[n]}\gamma_{ik}^{(t)}}
          $$

          $$
            \pi_k^{(t+1)}=\dfrac{\sum_{i\in[n]}\gamma_{ik}^{(t)}}{n}
          $$

  3. Assign the cluster:

      - compute $\gamma_{ik}^{(T+1)}$ , $\forall i \in [n]$
      - $x_i$ is assigned to cluster $\lambda_i$:
          $$
            \lambda_i=\argmax_{k=1,\cdots,K}\gamma_{ik}^{(T+1)}, \forall i \in [n]
          $$