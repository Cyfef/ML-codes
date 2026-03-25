import torch
from abc import ABC, abstractmethod
from tqdm import tqdm

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


class SVM:
    """
    Support Vector Machine (SVM) classifier using kernel methods
    """
    def __init__(self,
                 kernel_fn):
        """
        Args:
            kernel_fn: The kernel function to be used
        """
        self.K=kernel_fn
        self.b=None         
        self.alphas=None    #(n_support,1)
        self.support_vectors=None   #(n_support,d)
        self.support_labels=None    #(n_support,1)
        
    def predict(self,
                X:torch.Tensor
                ):
        """
        Predict the class labels and scores for the input data
        
        Args:
            X: Input samples tensor
        """

        kernel_matrix=self.K(X,self.support_vectors)
        scores=kernel_matrix @ (self.alphas*self.support_labels) + self.b
        preds=torch.where(scores>=0,1,-1)
        return scores,preds

    def fit(self, 
            X: torch.Tensor,    #(n,d)
            y: torch.Tensor,    #(n,1)
            C: float=1.0, 
            kkt_thr: float=1e-3, 
            max_passes: int=1000
            ):
        """
        Train SVM using SMO

        Args:
            X: training data
            y: training labels
            C: soft-margin parameter
            kkt_thr: Threshold for KKT conditions
            max_passes: Maximum number of passes through the training data
        """

        X = X.float()
        y = y.float()

        n = X.shape[0]

        self.support_vectors = X
        self.support_labels = y

        self.alphas = torch.zeros((n, 1))
        self.b = 0.0

        # kernel
        K = self.K(X, X)   #(n,n)

        # error cache
        def compute_E(i):
            f_i = (self.alphas * y).T @ K[:, i:i+1] + self.b
            return f_i.item() - y[i].item()

        passes = 0
        pbar = tqdm(total=max_passes, desc="Training SVM")

        while passes < max_passes:
            num_changed_alphas = 0

            for i in range(n):
                E_i = compute_E(i)

                if ((y[i] * E_i < -kkt_thr and self.alphas[i] < C) or
                    (y[i] * E_i > kkt_thr and self.alphas[i] > 0)):

                    E_list = torch.tensor([compute_E(k) for k in range(n)])
                    diff = torch.abs(E_list - E_i)
                    j = torch.argmax(diff).item()

                    if j == i:
                        continue

                    E_j = compute_E(j)

                    # ===== SMO update（你修好的那段）=====
                    alpha_i_old = self.alphas[i].clone()
                    alpha_j_old = self.alphas[j].clone()

                    y_i = y[i].item()
                    y_j = y[j].item()

                    if y_i != y_j:
                        L = max(0.0, alpha_j_old.item() - alpha_i_old.item())
                        H = min(C, C + alpha_j_old.item() - alpha_i_old.item())
                    else:
                        L = max(0.0, alpha_i_old.item() + alpha_j_old.item() - C)
                        H = min(C, alpha_i_old.item() + alpha_j_old.item())

                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alphas[j] = alpha_j_old - y_j * (E_i - E_j) / eta
                    self.alphas[j] = torch.clamp(self.alphas[j], L, H)

                    if abs(self.alphas[j].item() - alpha_j_old.item()) < 1e-5:
                        continue

                    self.alphas[i] = alpha_i_old + y_i * y_j * (alpha_j_old - self.alphas[j])

                    b1 = self.b - E_i \
                        - y_i * (self.alphas[i] - alpha_i_old) * K[i, i] \
                        - y_j * (self.alphas[j] - alpha_j_old) * K[i, j]

                    b2 = self.b - E_j \
                        - y_i * (self.alphas[i] - alpha_i_old) * K[i, j] \
                        - y_j * (self.alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alphas[i] < C:
                        self.b = b1.item()
                    elif 0 < self.alphas[j] < C:
                        self.b = b2.item()
                    else:
                        self.b = ((b1 + b2) / 2).item()

                    num_changed_alphas += 1

            # ===== tqdm 更新 =====
            if num_changed_alphas == 0:
                passes += 1
                pbar.update(1)
            else:
                passes = 0

            # 动态显示信息（非常有用）
            pbar.set_postfix({
                "changed": num_changed_alphas,
                "support": int((self.alphas > 1e-6).sum().item())
            })

        pbar.close()

        # support vectors
        mask = self.alphas.view(-1) > 1e-6

        self.support_vectors = X[mask]
        self.support_labels = y[mask]
        self.alphas = self.alphas[mask]




