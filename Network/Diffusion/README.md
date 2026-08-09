# Diffusion Model (DDPM)

## Settings

| symbol | shape | meaning |
| :--- | :--- | :--- |
| $d$ | $\mathbb{R}$ | Input dimension |
| $T$ | $\mathbb{R}$ | Number of diffusion steps (Time horizon) |
| $x_0$ | $\mathbb{R}^d$ | Original data point |
| $p_{\mathrm{data}}$ | - | Distribution of training data |
| $x_t$ | $\mathbb{R}^d$ | Latent variable at time $t$ |
| $\beta_t$ | $\mathbb{R}$ | Noise schedule |
| $\alpha_t$ | $\mathbb{R}$ | $1 - \beta_t$ |
| $\bar{\alpha}_t$ | $\mathbb{R}$ | $\prod_{s=1}^t \alpha_s$ |
| $\epsilon_\theta(x_t, t)$ | - | Function that predicts noise |
| $\phi(\cdot;\mu,\Sigma)$ | - | $\mathrm{P.D.F.}$ of $N(\mu,\Sigma)$ |
| $\theta$ | - | Parameter to be learned |
| $q(x_{1:T}\mid x_0)$ | - | Variational distribution approximating $p_\theta(x_{1:T}\mid x_0)$ |


## Formulas

- Reverse Process
  
    Starting from pure noise $x_T \sim N(0, I)$ and gradually denoising:

    $$
         p_\theta(x_{0:T})
        = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t),
    $$
    - **Learned reverse transition**:
        $$p_\theta(x_{t-1} \mid x_t) = \phi(x_{t-1};\mu_\theta(x_t, t), \Sigma_\theta(x_t,t))$$
    
    - ELBO decomposition:

        $$
            \log p_\theta(x_0)
        = \mathcal L_{\mathrm{ELBO}}(x_0;\theta)
        + \mathrm{KL}\left(q(x_{1:T}\mid x_0)\,\|\,p_\theta(x_{1:T}\mid x_0)\right)
        $$

        $$
            \mathcal L_{\mathrm{ELBO}}(x_0;\theta)
            := \mathbb E_{q(x_{1:T}\mid x_0)}\!\left[
                \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
            \right]
        $$

    - **DDPM mean function**:
        $$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

- Forward Process
  
    The data is gradually corrupted with Gaussian noise through a Markov chain:

    $$
        q(x_{1:T}\mid x_0) = \prod_{t=1}^{T} q(x_t\mid x_{t-1})
    $$

    - **Single-step transition**:
        $$q(x_t \mid x_{t-1}) = \phi(\sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I),\quad \alpha_t \in (0,1)$$
    - **Reparameterization**:
        $$
            x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_t, \quad \epsilon_t \sim N(0, I)
        $$

        $$
            x_t = \sqrt{\bar{\alpha_t}}x_0 + \sqrt{1-\bar{\alpha_t}}\epsilon, \quad \epsilon \sim N(0, I)
        $$
    - **Marginal distribution at arbitrary time**:
        $$q(x_t \mid x_0) = \phi(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$
    - Gaussian DDPM posterior:
        $$
            q(x_{t-1}\mid x_t,x_0)
            = \phi\bigl(x_{t-1};\tilde\mu_t(x_t,x_0),\,\tilde\beta_t I\bigr)
        $$

        where 

        $$
            \tilde\beta_t
            := \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
        $$

        $$
            \tilde\mu_t(x_t,x_0)
            :=
            \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\,x_0
            + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t
        $$

- Training objective
 
    - ELBO DDPM form:

        $$
             \mathcal L_{\mathrm{ELBO}}(x_0;\theta)
            =
            -\mathrm{KL}\!\left(q(x_T\mid x_0)\|p(x_T)\right)
            + \mathbb E_q\!\left[\log p_\theta(x_0\mid x_1)\right]
            - \sum_{t=2}^{T}\mathbb E_q\!\left[
                \mathrm{KL}\!\left(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t)\right)
            \right]
        $$

    - **DDPM loss**:
      
      A regression task that predicts the injected noise $\epsilon$:
        $$
        \mathcal{L}(\theta) = \mathbb E_{x_0\sim p_{\mathrm{data}}}
            \mathbb E_{t\sim\mathrm{U}(\{1,\dots,T\})}
            \mathbb E_{\varepsilon\sim N(0,I)}
            \left[
                \left\|
                    \varepsilon
                    - \varepsilon_\theta\!\left(
                        \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\ t
                    \right)
                \right\|_2^2
            \right]
        $$

- Training and Sampling Procedures

  1. **Training**:
     - Sample $x_0$ from the dataset.
     - Randomly choose a time step $t \in \{1, \dots, T\}$.
     - Sample noise $\epsilon \sim N(0, I)$ and construct $x_t$.
     - Optimize the loss $\mathcal{L}(\theta)$ to update the noise prediction model $\epsilon_\theta$.

  2. **Sampling (Generation)**:
     - Initialize $x_T \sim \mathcal{N}(0, I)$.
     - Iterate $t = T, \dots, 1$:
       - Compute the mean $\mu_\theta(x_t, t)$.
       - Sample $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$, where $z \sim N(0, I)$ , $\sigma_t^2=\beta_t$ or $\tilde\beta_t$ .
     - Output the final $x_0$.