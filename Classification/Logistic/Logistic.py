import torch

class Logistic_Regression():
    '''
    Logistic Regression
    '''
    def __init__(self):
        self.w_hat=None     #(d+1,1)
        self.lr=None

    def predict(
            self,
            X_0:torch.tensor    #(n,d)
    ) -> torch.tensor :
        '''
        Prediction

        Args:
            X_0:the input data to be predicted
        
        Returns:
            y_hat:the predicted classification label
        '''

        n=X_0.shape[0]
        ones=torch.ones(n,1)
        X=torch.cat((X_0,ones),dim=1)
        y_hat=torch.sigmoid(X @ self.w_hat)
        y_pred=(y_hat>0.5).int()
        return y_pred    #(n,1)
    
    def MLEloss(
            self,
            X_0:torch.tensor,   #(n,d)
            y:torch.tensor,     #(n,1)
    ) -> torch.tensor:
        '''
        Calculate MLE loss

        Args:
            X_0:the input training data 
            y:the gt classification labels

        Returns:
            loss:the MLE loss 
        '''

        n=X_0.shape[0]
        ones=torch.ones(n,1)
        X=torch.cat((X_0,ones),dim=1)
        loss=-self.w_hat.T @ X.T @ y + torch.sum(torch.log(torch.exp(self.w_hat.T @ X.T)+1))
        return loss

    def train(
            self,
            X_0:torch.tensor,   #(n,d)
            y:torch.tensor,     #(n,1)
            iterations:int,     
            lr:float,
    ) -> None:
        '''
        Training (gradient descent)
        
        Args:
            X_0:the input training data
            y:the gt classification labels
            iterations:the number of training iterations
            lr:learning rate of w_hat
        '''

        self.lr=lr

        n=X_0.shape[0]
        d=X_0.shape[1]

        ones=torch.ones(n,1)
        X=torch.cat((X_0,ones),dim=1)

        self.w_hat=torch.randn(d+1,1)

        for iteration in range(1,iterations+1):
            grad = -X.T @ y + X.T @ torch.sigmoid(X @ self.w_hat)    #(d+1,1)
            self.w_hat -= self.lr * grad

            if iteration % 10 == 0 :
                loss=self.MLEloss(X_0,y)
                print(f"[iter {iteration}] loss:{loss.item()}")             
