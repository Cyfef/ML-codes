import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class Base_kernel(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self,
                 x:torch.Tensor,    #(n1,d)
                 z:torch.Tensor     #(n2,d)
                 )->torch.Tensor:
        """
        Base kernel function.

        Args:
            x: First input data tensor
            z: Second input data tensor
        """
        ...

class Linear_kernel(Base_kernel):
    def __init__(self):
        super().__init__()

    def __call__(self,
                 x:torch.Tensor,    #(n1,d)
                 z:torch.Tensor     #(n2,d)
                 ):
        y=x@z.T
        return y    #(n1,n2)

class Polynomial_kernel(Base_kernel):
    def __init__(self,
                 degree:int,
                 c:float
                 ):
        super().__init__()
        self.p=degree
        self.c=c

    def __call__(self,
                 x:torch.Tensor,    #(n1,d)
                 z:torch.Tensor     #(n2,d)
                 ):
        y=(x@z.T+self.c)**self.p
        return y    #(n1,n2)
        
class RBF_kernel(Base_kernel):
    def __init__(self,
                 sigma:float
                 ):
        super().__init__()
        self.sigma=sigma

    def __call__(self,
                 x:torch.Tensor,    #(n1,d)
                 z:torch.Tensor     #(n2,d)
                 ):
        x_norm=torch.sum(x**2,dim=1,keepdim=True)   #(n1,1)
        z_norm=torch.sum(z**2,dim=1,keepdim=True)   #(n2,1)

        dot_product=x@z.T
        l2_dists=x_norm+z_norm.T-2*dot_product

        y=torch.exp((-1.0/(2*self.sigma**2))*l2_dists)
        return y    #(n1,n2)


class GP(nn.Module):
    def __init__(self, kernel="rbf"):
        super().__init__()
        
        # ===== kernel hyperparameters =====
        self.log_sigma_f = nn.Parameter(torch.tensor(0.0))  # log σ_f
        self.log_ell = nn.Parameter(torch.tensor(0.0))      # log ℓ
        
        # noise variance σ^2
        self.log_sigma_n = nn.Parameter(torch.tensor(-1.0))
        
        self.kernel_type = kernel

    # ===== kernel function k(x, x') =====
    def kernel(self, X1, X2):
        """
        X1: (n, d)
        X2: (m, d)
        return: (n, m)
        """
        sigma_f = torch.exp(self.log_sigma_f)
        ell = torch.exp(self.log_ell)
        
        if self.kernel_type == "rbf":
            # ||x - x'||^2
            dist = torch.cdist(X1, X2) ** 2
            return sigma_f**2 * torch.exp(-0.5 / ell**2 * dist)
        
        elif self.kernel_type == "linear":
            return X1 @ X2.T
        
        else:
            raise NotImplementedError

    # ===== compute K + σ^2 I =====
    def compute_covariance(self, X):
        K = self.kernel(X, X)
        sigma_n = torch.exp(self.log_sigma_n)
        n = X.shape[0]
        return K + sigma_n**2 * torch.eye(n)

    # ===== log marginal likelihood =====
    def log_marginal_likelihood(self, X, y):
        """
        y: (n, 1)
        """
        K = self.compute_covariance(X)
        
        L = torch.linalg.cholesky(K)
        
        # solve K^{-1}y
        alpha = torch.cholesky_solve(y, L)
        
        term1 = -0.5 * y.T @ alpha
        term2 = -torch.sum(torch.log(torch.diag(L)))
        term3 = -0.5 * X.shape[0] * torch.log(torch.tensor(2 * torch.pi))
        
        return (term1 + term2 + term3).squeeze()

    # ===== prediction =====
    def predict(self, X, y, X_star):
        K = self.compute_covariance(X)
        L = torch.linalg.cholesky(K)
        
        # k_*
        K_star = self.kernel(X, X_star)        # (n, m)
        K_star_star = self.kernel(X_star, X_star)  # (m, m)
        
        # mean
        alpha = torch.cholesky_solve(y, L)
        mu_star = K_star.T @ alpha
        
        # variance
        v = torch.linalg.solve(L, K_star)
        sigma_star = K_star_star - v.T @ v
        
        return mu_star, sigma_star