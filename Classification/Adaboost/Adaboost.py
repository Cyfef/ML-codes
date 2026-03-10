import torch
from enum import Enum
from typing import List

class PurityMeasure(Enum):
    IG="Information Gain"
    GR="Information Gain Ratio"
    GINI="Gini Index"

class WeakClassifier:
    '''
    Weak Classifier: a decision stump
    '''
    def __init__(
            self,
            measure_type:PurityMeasure=PurityMeasure.GINI
    ):
        '''
        Args:
            measure_type: 
        '''
        self.feature=None
        self.tau=None

        self.low_pred=None
        self.high_pred=None

        self.alpha=None
        self.measure_type=measure_type

    def fit(
            self,
            X:torch.Tensor,             #(n,d)
            y:torch.Tensor,             #(n,1)
            weights:torch.Tensor,       #(n,)
    )->None:
        '''
        let model fit the training data 

        Args:
            X:training data
            y:gt labels
            weights: sample weights
        '''
        n,d=X.shape

        best_feature=None
        beat_tau=None
        best_gini=float('inf')

        for A in range(d):
            val_types=torch.unique(X[:,A])    #(m,)
            m=val_types.shape[0]

            if m<=1:        # no thresholds to fit
                continue

            taus=(val_types[:-1] + val_types[1:]) / 2.0
            for tau in taus:
                mask=X[:, A] > tau

                low_weights=weights[~mask]
                high_weights=weights[mask]
                low_y=y[~mask]
                high_y=y[mask]

                low_gini=self.Gini_weighted(low_y,low_weights)
                high_gini=self.Gini_weighted(high_y,high_weights)

                gini_mid_A=low_weights.sum()*low_gini+high_weights.sum()*high_gini
                gini_mid_A=gini_mid_A.item()
                
                if gini_mid_A<best_gini:
                    best_gini=gini_mid_A
                    best_feature=A
                    best_tau=tau.item()
                    self.low_pred=self.majority_label(low_y,low_weights)
                    self.high_pred=self.majority_label(high_y,high_weights)

        if best_feature is None:
            best_feature=0
            best_tau=0.0
            self.high_pred=self.low_pred=self.majority_label(y,weights)

        self.feature,self.tau=best_feature,best_tau
    
    def predict(
            self,
            X:torch.Tensor,     #(m,d)
    )->torch.Tensor:
        '''
        Args:
            X: the data to be predicted
        '''
        X_A=X[:,self.feature]
        y_pred=torch.where(X_A>self.tau,self.high_pred,self.low_pred)
        return y_pred.unsqueeze(1)    #(m,)   
        
    @staticmethod
    def Gini_weighted(
        y:torch.Tensor,         #(n,1)
        weights:torch.Tensor,   #(n,)
    )->float:
        '''
        Weighted Gini index 

        Args:
            y:the labels vector
            weights:the label weights
        '''
        pos_weight = weights[(y == 1).squeeze()].sum()
        neg_weight = weights[(y == -1).squeeze()].sum()
        total = pos_weight + neg_weight
        gini = 1.0 - (pos_weight/total)**2 - (neg_weight/total)**2
        return gini
    
    @staticmethod
    def majority_label(
        y:torch.Tensor,         #(n,1)
        weights:torch.Tensor,   #(n,)
    )->int:
        """
        Return the class (+1 or -1) with the largest total weight
        
        Args:
            y:the labels vector
            weights:the label weights
        """
        pos_weight = weights[(y == 1).squeeze()].sum()
        neg_weight = weights[(y == -1).squeeze()].sum()
        return 1 if pos_weight >= neg_weight else -1

class Adaboost:
    def __init__(
            self,
            T:int,
    ):
        '''
        Args:
            T:number of weak classifiers/iterations
        '''
        self.T=T
        self.clfs=[]

    def fit(
            self,
            X:torch.Tensor,     #(n,d)
            y:torch.Tensor,     #(n,)
    )->None:
        n,d=X.shape
        weights=torch.ones(n)/n

        # train each weak classifier
        for _ in range(self.T):
            f_t=WeakClassifier()
            f_t.fit(X,y,weights)

            y_pred_t=torch.tensor(f_t.predict(X)).squeeze()
            eps_t=weights[y_pred_t!=y].sum()

            if eps_t>0.5:
                break
            
            alpha_t=0.5 * torch.log((1-eps_t)/eps_t)
            f_t.alpha=alpha_t

            weights_bar=weights*torch.exp(-alpha_t*y*y_pred_t)
            weights=weights_bar/weights_bar.sum()

            self.clfs.append(f_t)
    
    def predict(
            self,
            X:torch.Tensor      #(m,d)
    )->List:
        m=X.shape[0]
        preds=torch.zeros(m)

        for f in self.clfs:
            y_pred=f.predict(X)
            preds+=f.alpha*y_pred

        preds=torch.sign(preds)
        return preds.tolist()

    