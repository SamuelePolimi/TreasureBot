#This program should be called from the main folder as following
#
#python -m examples.lstm_example
import numpy as np
from optimizer.lstm import LSTMNet
from core.bot import Bot

print "Loading dataset"

train_set = np.load("dataset/forex/ds2/train_set.npy")
validation_set = np.load("dataset/forex/ds2/validation_set.npy")

#very short sequence for debugging purpose

train_set = train_set[:,:,:]
validation_set = validation_set[:,:,:]

n_series = 1
n_features = 2    #timing & volume

config = {
    #Shape (2,10) means  2 layers of 10 neurons
    "sharedBoxShape" : (4, 10),
    "blocksShape": (5,10),
    "nLSTMCells": 10,
    "decisionBlockShape": (3,10),
    "dropout": 1.,
    "batch_size": 10
}

print "Model configuration"

opt = LSTMNet(config, train_set, validation_set, n_series, n_features)

print "Learning started"
for i in xrange(0,1):
    out_ = opt.learn()
    print "Epoch" , i, ":", "train gain:", out_[0], "validation gain:", out_[1]

suggester = opt.finalize()

bot = Bot(suggester,n_series,n_features,False,1000, 0.1)
capital = []
for i in range(validation_set.shape[0]):
    bot.reset()
    for t in range(validation_set.shape[1]): 
        bot.step(validation_set[i,t,:])
    capital.append(bot.getVirtualCapital())

print "you gained:", (sum(capital) /( len(capital) + 0.0))
    
    