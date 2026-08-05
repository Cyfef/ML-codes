# Variational Autoencoder (VAE)

## Settings

| symbol | shape | meaning | 
| :--- | :--- | :--- | 
| $n$ | $\mathbb{R}$ | the number of samples |
| $d$ | $\mathbb{R}$ | input dimension |   
| $k$ | $\mathbb{R}$ | latent dimension |
| $x^{(i)}$ | $\mathbb{R}^d$ | an unlabeled data point |
| $z$ | $\mathbb{R}^k$ | the latent variable |
| $p(z)$ | - | a simple prior of the latent variable |
| $\phi(\cdot;\mu,\Sigma)$ | - | the $\mathrm{p.d.f.}$ of $N(\mu,\Sigma)$ |
| $\theta$ | - | the parameter of the conditional likelihood (decoder) |
| $\theta'$ | - | the parameter of the variational distribution (encoder) |
| $p(x\mid z;\theta)$ | - | conditional likelihood |
| $q(z\mid x;\theta')$ | - | variational distribution approximating $p(z\mid x;\theta)$ |
| $\sigma^2$ | - | a fixed hyperparameter of Gaussian distribution |


## Formulas

- Generative Model
    
    1. Prior:

        $$
            z \sim N(0,I)
        $$
    2. Conditional likelihood:

        $$
            p(x\mid z;\theta)=\phi(x;f_\theta(z),\sigma^2I)
        $$

        where $f_\theta:\mathbb{R}^k\to\mathbb{R}^d$ is the mean function 

    3. Marginal likelihood:

        $$
            p(x;\theta)=\int_z p(x\mid z;\theta)p(z)\mathrm dz
        $$

- MLE objective (intractable):
  
  $$
    \argmax_\theta\sum_{i=1}^n \log p(x_i;\theta)
  $$

- Inference Model:

    $$
        q(z\mid x;\theta')=\phi\bigl(z; \mu_{\theta'}(x),\Sigma_{\theta'}(x)\bigr)
    $$

    $$
        \Sigma_{\theta'}(x)=\mathrm{diag}(\sigma_{\theta'}^2(x))
    $$

- ELBO decomposition:

    $$
     \log p(x;\theta)
    =
    \underbrace{
    \int_z q(z\mid x;\theta')\log p(x\mid z;\theta)\mathrm{d}z
    -
    \mathrm{KL}\left(q(z\mid x;\theta')\middle\|p(z)\right)
    }_{\mathrm{ELBO}}
    +
    \underbrace{
    \int_z q(z\mid x;\theta') \log \dfrac{q(z\mid x;\theta')}{p(z\mid x;\theta)}\mathrm{d}z
    }_{\mathrm{KL}\left(q(z\mid x;\theta')\,\middle\|\,p(z\mid x;\theta)\right)\ge 0}
    $$

- Reparameterization Trick

    $$
        q(z\mid x;\theta')=\phi\bigl(z; \mu_{\theta'}(x),\,\Sigma_{\theta'}(x)\bigr), \quad \Sigma_{\theta'}(x)=\mathrm{diag}(\sigma_{\theta'}^2(x))
    $$

    $$
        z=\mu_{\theta'}(x)+L_{\theta'}(x)\varepsilon=\mu_{\theta'}(x)+\sigma_{\theta'}(x)\odot \varepsilon,\quad \varepsilon \sim N(0,I)
    $$

    where $L_{\theta'}(x)L_{\theta'}(x)^\top=\Sigma_{\theta'}(x)$

- Gaussian-Gaussian KL:

    $$
        \mathrm{KL}(q\|p)
    =
    \frac{1}{2}\left(
        \mathrm{tr}(\Sigma) + \mu^\top \mu - k - \log\det(\Sigma)
    \right)
    $$

    where $q=\phi(z;\mu,\Sigma_{k\times k})$ , $p=\phi(z;0,I)$

- Training process:

    1. Initialize parameters $\theta$ (decoder) and $\theta'$ (encoder)
    2. Repeat until convergence:
        
        For each mini‑batch ${x^{(1)},\dots,x^{(m)}}$ drawn from the dataset:
        - Encode: compute $\mu_{\theta'}(x^{(i)})$ and $\sigma_{\theta'}(x^{(i)})$ 
        - Reparameterize: sample $\varepsilon^{(i)}\sim N(0,I)$ and set $z^{(i)} = \mu_{\theta'}(x^{(i)}) + \sigma_{\theta'}(x^{(i)})\odot \varepsilon^{(i)}$ 
        - Decode: compute $f_\theta(z^{(i)})$ 
        - Compute loss (negative ELBO):

            $$
                \mathcal{L} = \dfrac{1}{m}\sum_{i=1}^m\left[
                -\log p\bigl(x^{(i)}\mid z^{(i)};\theta\bigr)
                + \mathrm{KL}\bigl(q(z\mid x^{(i)};\theta')\,\|\,p(z)\bigr)
                \right]
            $$
        - Backpropagate gradients w.r.t. $\theta$ and $\theta'$, and update parameters 
