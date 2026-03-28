import torch

class LassoRegression:
    '''Lasso Regression'''
    def __init__(self,
                 lam:float):
        '''
        Args:
            lam: hyperparameter for regularization term
        '''
        self.w_hat=None     #(d+1,1)
        self.lam=lam
        self.lr=None
        
    def train(
            self,
            X_0:torch.Tensor,   #(n,d)
            y:torch.Tensor,     #(n,1)
            iterations:int=1000,
            lr:float=0.01
    ) -> None :
        '''
        Train the model
        
        Args:
            X_0: input training data
            y: gt label
            iterations: number of training iterations
            lr: learning rates
        '''
        n,d = X_0.shape
        ones = torch.ones(n, 1)
        X = torch.cat((X_0, ones), dim=1)   #(n,d+1)

        self.lr=lr
        self.w_hat = torch.zeros(d + 1, 1)

        for iter in range(iterations):
            y_hat = X @ self.w_hat
            
            #loss
            mse = ((y - y_hat) ** 2).mean()
            l1 = torch.abs(self.w_hat).sum()   
            loss = mse + self.lam * l1

            print(f"Iter {iter} | Loss: {loss.item():.6f}")

            y_hat = X @ self.w_hat   #(n,1)

            grad = -(2/n) * X.T @ (y - y_hat)
            reg = torch.sign(self.w_hat)
            grad = grad + self.lam * reg

            self.w_hat = self.w_hat - lr * grad


    def predict(
            self,
            X_0:torch.Tensor    #(n,d)
    ) -> torch.Tensor :
        '''
        Prediction
        
        Args:
            X_0:the input data to be predicted
        
        Returns:
            y_hat:the predicted value
        '''

        n=X_0.shape[0]
        ones=torch.ones(n,1)
        X=torch.cat((X_0,ones),dim=1)
        y_hat=X @ self.w_hat
        return y_hat    #(n,1)
        
        