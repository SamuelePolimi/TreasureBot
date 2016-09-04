class Optimizer:     
        
    """
    OptimizerConfig: dict with all configuration of the optimizer
    train_set: 3D numpy array to train the model
    validation_set: 3D numpy array with the purpose to evaluate the model (-- train_set and validation_set should be disjoint --)
    n_series: number of stock took in consideration (and therefore to buy and sell)
    features: number of additional data to use for the decision
    """
    
    def __init__(optimizerConfig, train_set, validation_set, n_series, features):
        pass
    
    """It will return a tuple with (gain on train_set, gain on validation_set)"""
    def learn(self):
        raise("Not implemented")

    
    def finalize(self):
        raise("Not implemented")

