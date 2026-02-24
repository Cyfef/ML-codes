import torch
from enum import Enum

class PurityMeasure(Enum):
    IG="Information Gain"
    GR="Information Gain Ratio"
    GINI="Gini Index"

class WeakClassifier:
    '''
    Weak Classifier: decision stump
    '''
    def __init__(
            self,
            alpha,
            measure_type:PurityMeasure
    ):
        self.tree=None
        self.alpha=None
        self.measure_type=measure_type

    def train(
            self,
            X:torch.tensor,
    ):
        pass

    @staticmethod
    def H(
            y:torch.tensor      #(n,1)
    )->float:
        '''
        Calculate the information entropy of the dataset with respect to its empirical label distribution

        Args:
            y:training labels
        '''
        label_types, label_counts = torch.unique(y, return_counts=True)     #(K,),(K,)
        n=y.shape[0]
        entropy=-torch.sum((label_counts/n)*torch.log2(label_counts/n))
        return entropy

    @classmethod
    def H_mid_A(
            cls,
            X:torch.tensor,     #(n,d)
            y:torch.tensor,     #(n,1)
            A:int
    )->float:
        '''
        Calculate the conditional information entropy of dataset given the feature

        Args:
            X:training dataset
            y:training labels
            A:the feature index(0~d-1)
        '''
        val_types, val_counts=torch.unique(X[:,A], return_counts=True)    #(m,),(m,)
        n=X.shape[0]
        ent_fea=[]
        for val in val_types:
            mask=(X[:,A]==val)
            y_val=y[mask]
            ent_fea.append(cls.H(y_val))
        ent_fea=torch.tensor(ent_fea)
        entropy=torch.sum(ent_fea*(val_counts/n))
        return entropy

    @staticmethod
    def H_and_A(
            X:torch.tensor,     #(n,d)
            A:int
    )->float:
        '''
        Calculate the information entropy of a feature in dataset
        
        Args:
            X:training dataset
            A:the feature index(0~d-1)
        '''
        val_types, val_counts=torch.unique(X[:,A], return_counts=True)    #(m,),(m,)
        n=X.shape[0]
        entropy=-torch.sum((val_counts/n)*torch.log2(val_counts/n))
        return entropy
    
    @staticmethod
    def Gini(
        X:torch.tensor,     #(n,d)
        y:torch.tensor,     #(n,1)
    ):
        label_types, label_counts = torch.unique(y, return_counts=True)     #(K,),(K,)
        n=X.shape[0]
        return 1-torch.sum((label_counts/n)**2)
        
    @classmethod
    def Gini_mid_A(
        cls,
        X:torch.tensor,     #(n,d)
        y:torch.tensor,     #(n,1)
        A:int,
    ):
        val_types, val_counts=torch.unique(X[:,A], return_counts=True)    #(m,),(m,)
        n=X.shape[0]
        gini_fea=[]
        for val in val_types:
            mask=(X[:,A]==val)
            X_val=X[mask]
            y_val=y[mask]
            gini_fea.append(cls.Gini(X_val,y_val))
        gini_fea=torch.tensor(gini_fea)
        gini=torch.sum(gini_fea*(val_counts/n))
        return gini
        
    def cal_measure(
            self,
            X:torch.tensor,     #(n,d)
            y:torch.tensor,     #(n,1)
            A:int,
    ):
        '''
        Calculate the purity measure of a feature

        Args:
            X:training dataset
            y:training labels
            A:the feature index(0~d-1)
        '''
        if self.measure_type==PurityMeasure.IG:
            return self.H(y)-self.H_mid_A(X,y,A)
        elif self.measure_type==PurityMeasure.GR:
            denominator = self.H_and_A(X, A)
            if denominator == 0:
                return 0.0  
            return (self.H(y) - self.H_mid_A(X, y, A)) / denominator
        elif self.measure_type==PurityMeasure.GINI:
            return -self.Gini_mid_A(X,y,A)


class Adaboost:
    def __init__(
            self,
            
    ):
        pass

    