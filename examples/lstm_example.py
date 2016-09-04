#This program should be called from the main folder as following
#
#python -m examples.lstm_example
import numpy as np
from optimizer.lstm import LSTMNet

print "Loading dataset"

""" If you don't want to try with long series (space and time consuming), try this
train_set = np.load("dataset/forex/ds1/train_set.npy")[:,:10,:]
validation_set = np.load("dataset/forex/ds1/validation_set.npy")[:,:10,:]"""


train_set = np.load("dataset/forex/ds1/train_set.npy")
validation_set = np.load("dataset/forex/ds1/validation_set.npy")

n_series = 1
n_features = 2    #timing & volume

config = {
    #Shape (2,10) means  2 layers of 10 neurons
    "sharedBoxShape" : (3, 10),
    "blocksShape": (4,5),
    "nLSTMCells": 20,
    "decisionBlockShape": (2,10),
    "dropout": 1.,
    "batch_size": 10
}

print "Model configuration"

opt = LSTMNet(config, train_set, validation_set, n_series, n_features)

print "Learning started"
for i in xrange(0,100):
    print "Epoch" , i, ":", opt.learn()