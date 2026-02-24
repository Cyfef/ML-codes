import torch
from enum import Enum

class PurityMeasure(Enum):
    IG="Information Gain"
    GR="Information Gain Ratio"
    GINI="Gini Index"

class WeakClassifier:
    '''
    Weak Classifier: a one-layer decision tree
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


class Adaboost:
    def __init__(
            self,
            
    ):
        pass

    