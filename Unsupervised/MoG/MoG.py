import torch


class MoG:
    """
    Mixture of Gaussians
    """

    def __init__(self,
                 K:int,
                 iterations:int=100,
                 ) -> None:
        """
        Args:
            K: the number of clusters
            iterations: the iterations of optimization
        """
        self.K=K
        self.T=iterations
        self.pis=None
        self.mus=None
        self.Sigmas=None
        self.gammas=None

    def fit(self,
            X:torch.Tensor      #(n,d)
            ):
        """
        EM optimization

        Args:
            X: training data
        """
        n=X.shape[0]

        #init
        self.pis=torch.full((self.K,),1/self.K)     #(K,)
        indices = torch.randperm(n)[:self.K]
        self.mus=X[indices]         #(K,d)
        self.Sigmas=torch.eye(X.shape[1]).unsqueeze(0).repeat(self.K,1,1)

        #iter
        for _ in range(self.T):
            # E step
            self.E_step(X)
            # M step
            self.M_step(X)

    def predict(self,
                X:torch.Tensor
                ):
        self.E_step(X)
        return torch.argmax(X, dim=1)   #(n,)
    
    def E_step(self,
                  X:torch.Tensor    #(n,d)
                  ):
        log_probs = []  # log pi_k + log pdf

        for k in range(self.K):
            N_distribution = torch.distributions.MultivariateNormal(
                loc=self.mus[k],
                covariance_matrix=self.Sigmas[k]
            )
            log_prob = N_distribution.log_prob(X) + torch.log(self.pis[k])  # (n,)
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs, dim=1)       # (n, K)

        # log-sum-exp 
        max_log = torch.max(log_probs, dim=1, keepdim=True)[0]   # (n,1)
        probs = torch.exp(log_probs - max_log)
        self.gammas = probs / probs.sum(dim=1, keepdim=True)     # (n,K)

    def M_step(self,
                X:torch.Tensor
                ):
        n, _ = X.shape
        Nk = self.gammas.sum(dim=0)         # (K,)

        self.pis = Nk / n                   # (K,)
        self.mus = (self.gammas.T @ X) / Nk.unsqueeze(1)
 
        X_centered = X.unsqueeze(1) - self.mus.unsqueeze(0)     # (n,K,d)
        outer = X_centered.unsqueeze(3) * X_centered.unsqueeze(2)      # (n,K,d,d)
        self.Sigmas = (self.gammas.unsqueeze(2).unsqueeze(3) * outer).sum(dim=0) / Nk.view(-1,1,1)  #(K,d,d)




