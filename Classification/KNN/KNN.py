import torch
from typing import List

class KNN:
    '''
    K Nearest Neighbor
    '''
    def __init__(
            self,
            K:int,
    ):
        '''
        Args:
            K:the number of neighbors to be considered
        '''
        self.K=K

    def load_data(
            self,
            X:torch.tensor,     #(n,d)
            y:torch.tensor,     #(n,) 
    ):
        '''
        Load the data and gt labels

        Args:
            X:the samples data
            y:the gt labels
        '''
        self.X=X
        self.y=y

    def predict(
            self,
            X:torch.tensor      #(m,d)
    )->List:
        '''
        Predict labels for test samples

        Args:
            X: input test samples

        Returns:
            list of predicted labels 
        '''
        m=X.shape[0]

        y_pred=[]
        for i in range(m):
            X_i=X[i,:]
            dists=torch.sum((self.X-X_i)**2,dim=1)
            _, indices = torch.topk(dists, self.K, largest=False, sorted=True)
            labels_K=self.y[indices]
            unique_labels, counts = torch.unique(labels_K, return_counts=True)
            max_count_idx = torch.argmax(counts)
            y_pred.append(unique_labels[max_count_idx].item())
        return y_pred
