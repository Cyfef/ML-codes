import torch
import random

class KMeans:
    '''
    K-means clustering
    '''
    def __init__(
            self,
            K:int,
            iterations:int
    ):
        '''
        Args:
            K:
            iterations:
        '''
        self.K=K
        self.iterations=iterations
        self.centers=None

    def train(
            self,
            X:torch.tensor,     #(n,d)
    ):
        '''
        Perform K-means through iterations

        Args:
            X:data to be fit
        '''
        self.centers = X[random.sample(range(len(X)), self.K),:]
        for _ in range(self.iterations):
            # Assignment step
            assigns=torch.tensor(self.predict(X))
            # Update step
            new_centers=[]
            for k in range(self.K):
                mask=(assigns==k)
                X_k=X[mask,:]

                if X_k.size(0)==0:
                    # empty
                    new_centers.append(self.centers[k])
                else:
                    means=X_k.mean(dim=0)   #(d,)
                    new_centers.append(means)
            self.centers=torch.stack(new_centers,dim=0)

    def predict(
            self,
            X:torch.tensor,     #(n,d)
    ):
        '''
        Predict the assignments/labels of the given data

        Args:
            X:data to be fit
        '''
        diffs=X[:,None,:]-self.centers[None,:,:]    #(n,K,d)
        dists=torch.sum(diffs**2,dim=2)     #(n,K)
        y_pred=torch.argmin(dists,dim=1)    #(n,)
        return y_pred.tolist()
        

    





