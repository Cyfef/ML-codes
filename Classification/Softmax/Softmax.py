import torch

class SoftmaxRegression:
    '''
    Softmax Regression classifier
    '''
    def __init__(self):
        self.W=None     #(K,d)
        self.b=None     #(K,1)
        self.lr=None

    def train(
            self,
            X:torch.Tensor,     #(n,d)
            Y:torch.Tensor,     #(n,K)
            iterations:int=500,
            lr:float=0.1,
    )->None:
        '''
        Train the classifier

        Args:
            X: training data
            Y: training labels
            iterations: number of training iterations
            lr: learning rate
        '''
        self.lr=lr

        n,d=X.shape
        K=Y.shape[1]

        self.W=torch.zeros((K,d))
        self.b=torch.zeros((K,1))

        for iter in range(iterations):
            S_hat,_=self.predict(X)

            #loss
            loss=-torch.sum(Y*torch.log(S_hat))/n
            print(f"Iter {iter}: loss {loss.item():.6f}")

            grad_W=(S_hat-Y).T@X
            grad_W/=n

            grad_b = torch.mean(S_hat - Y, dim=0).view(K, 1)

            self.W-=self.lr*grad_W
            self.b-=self.lr*grad_b
        

    def predict(
            self,
            X:torch.Tensor      #(m,d)
    ):
        '''
        Prediction

        Args:
            X: data to be predicted
        '''
        Z_hat=X@self.W.T+self.b.T
        Z_hat = Z_hat - Z_hat.max(dim=1, keepdim=True)[0]
        S_hat=torch.softmax(Z_hat,dim=1)
        y_hat=torch.argmax(S_hat,dim=1)
        return S_hat,y_hat
    