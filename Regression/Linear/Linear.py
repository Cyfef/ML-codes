import torch

class LinearRegression():
    '''Linear Regression'''
    def __init__(self) -> None:
        self.w_hat=None     #(d+1,1)
        
    def closed_form_solution(
            self,
            X_0:torch.Tensor,   #(n,d)
            y:torch.Tensor      #(n,1)
    ) -> None :
        '''
        Calculate the closed-form solution
        
        Args:
            X_0:the input train data
            y:the gt label
        '''

        n=X_0.shape[0]
        ones=torch.ones(n,1)
        X=torch.cat((X_0,ones),dim=1)
        self.w_hat=torch.linalg.inv(X.T @ X) @ X.T @ y

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
        
        