import torch
from enum import Enum

class PurityMeasure(Enum):
    IG="Information Gain"
    GR="Information Gain Ratio"
    GINI="Gini Index"

class Decision_Trees:
    '''
    Decision Trees 
    '''
    def __init__(
            self,
            measure_type:PurityMeasure
    ):
        '''
        Args:
            measure_type:the type of purity measure to be used
        '''
        self.tree=None
        self.measure_type=measure_type

    def predict(
            self,
            X:torch.tensor      #(n,d)
    ):
        '''
        Predict using decision tree
        
        Args:
            x:the data to be predicted
        '''
        def _predict(node, x):
            # leaf
            if not isinstance(node, dict):
                return node
            # feature idx
            feat_idx = next(iter(node))
            # feature val
            feat_val = x[feat_idx].item()
            # subtree dict
            subtree = node[feat_idx]
            if feat_val in subtree:
                return _predict(subtree[feat_val], x)
            else:
                # unseen feature
                return None

        return [_predict(self.tree, X[i]) for i in range(X.shape[0])]
        
    def build_tree(
            self,
            X:torch.tensor,     #(n,d)
            y:torch.tensor,     #(n,1)
    ):
        '''
        Build the decision tree using Greedy Algorithm
        
        Args:
            X:training dataset
            y:training labels
        '''
        def _build(X, y):
            d = X.shape[1]

            # only 1 class
            if len(torch.unique(y)) == 1:
                return torch.unique(y)[0].item()   

            # each feature the same value, return the class label with the highest frequency
            flag = 0
            for i in range(d):
                if len(torch.unique(X[:, i])) > 1:
                    flag = 1
                    break
            if flag == 0:
                label_types, label_counts = torch.unique(y, return_counts=True)
                return label_types[torch.argmax(label_counts)].item()

            # find best feature to split data
            best_idx, best_tensors = self.find_best_feature(X, y)
            tree = {best_idx.item(): {}}

            # recursively
            for val, (X_sub, y_sub) in best_tensors.items():
                key = val.item() if isinstance(val, torch.Tensor) else val
                tree[best_idx.item()][key] = _build(X_sub, y_sub)

            return tree

        self.tree = _build(X, y)

    def find_best_feature(
            self,
            X:torch.tensor,
            y:torch.tensor,
    ):
        '''
        Find the best feature from current dataset

        Args:
            X:training dataset
            y:training labels
        '''
        d=X.shape[1]
        features_measure=torch.tensor([self.cal_measure(X,y,A) for A in range(d)])
        best_idx=torch.argmax(features_measure)

        delete_mask = torch.ones(d, dtype=torch.bool)
        delete_mask[best_idx] = False

        best_tensors={}     #dict
        val_types, val_counts=torch.unique(X[:,best_idx], return_counts=True)    #(m,),(m,)
        for val in val_types:
            mask=(X[:,best_idx]==val)
            X_val=X[mask]
            y_val=y[mask]
            X_val=X_val[:,delete_mask]
            best_tensors[val]=(X_val,y_val)
        
        return best_idx,best_tensors
    
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