class exponential_decay():
    def __init__(
            self, 
            initial_learning_rate:float=0.001, 
            decay_rate:float=0.5, 
            decay_epochs:int=100
        ):
        '''
        Exponential decay learning rate scheduler

        Args:
            initial_learning_rate: Initial learning rate at epoch 0
            decay_rate: Multiplicative factor of learning rate decay
            decay_epochs: Number of epochs between each decay step
        '''
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_epochs = decay_epochs

    def __call__(
            self, 
            epoch:int
        ):
        '''
        Compute the learning rate at a given epoch

        Args:
            epoch: Current training epoch (starting from 0)
        '''
        return self.initial_learning_rate * self.decay_rate ** (epoch // self.decay_epochs)