"""
This class will be used to train a regressor (with the method learn), and to produce the final object called "BOT" which will be used to trade stocks
"""
import numpy as np

class Learner:
    
    def __init__(self,regressor, train_set, validation_set):
        self.regressor = regressor
        self.train_set
        self.validation_set
        
        # n_row: number of sample in the dataset
        # serie_length: how long is a serie?
        # n_serie: how many stock are we learning?
        # features: how many values describe a serie?
        # (1st feature will be always the stock price | last features will be the cost to perform the long short action)
    
        n_row, serie_length, n_series, features = self.train_set.shape
        _, serie_length_v, n_series_v, features_v = self.train_set.shape
                
        assert serie_length==serie_length_v and n_series==n_series_v and features==features_v, "Trainset and validationset have different shape."
        
        # here we really build the regressor
        self.regressor.initialize(serie_length, n_series, features)
        
    """
    Perform a learning process. the return is the gain with the train_set and the validation_set
    """
    def learn(self):
        
        train_gain = self.regressor.learn(self.train_set)
        validation_gain = self.regressor.evaluate(self.validation_set)
        
        return (train_gain, validation_gain)
    
    """
    This method return an object to use as bot,
    """
    def getBot(self):
        final_object = self.regressor.finalize()
        bot = None #bot = Bot(final_object)
        return bot
        
        
              
              
        
          
