#This program should be called from the main folder as following
#
#python -m examples.lstm_example
import numpy as np
from optimizer.lstm import LSTMNet

print "Loading dataset"

train_set = np.load("dataset/forex/ds2/train_set.npy")
validation_set = np.load("dataset/forex/ds2/validation_set.npy")

#very short sequence for debugging purpose
"""
train_set = train_set[:,:50,:]
validation_set = validation_set[:,:50,:]"""

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
for i in xrange(0,500):
    out_ = opt.learn()
    print "Epoch" , i, ":", "train gain:", out_[0], "validation gain:", out_[1]