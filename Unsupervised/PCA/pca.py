import torch

class PCA:
    '''
    Principal Components Analysis
    '''
    def __init__(
            self,
            K:int
    )->None:
        '''
        Args:
            K:the number of principal components
        '''
        self.K=K
        
    def reduce(
            self,
            X:torch.tensor      #(n,d)
    )->torch.tensor:
        '''
        Reduce the dimension of the imput data

        Args:
            X:the imput data 
        '''
        d=X.shape[1]
        X_hat=X-X.mean(dim=0)
        Sigma=X_hat.T @ X_hat
        lambdas,U=torch.linalg.eigh(Sigma)
        indices = torch.arange(d - self.K, d)
        indices_desc = indices.flip(0)
        U_P = U[:, indices_desc]
        #U_P=U[:, -self.K:][:, ::-1]
        return X @ U_P
        