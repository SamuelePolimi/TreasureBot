"""----------------------------------------------------------------------------

Created on Fri Jul 29 15:53:23 2016

@author: samuele

This script should solve Stock exchange with an improvement of the Direct_rl with Long Short Time Memory cell [Hochriter 1991]
This should (partially) solve the problem of the gradient vanishing, leading to a better usage of the time dependencies


Reference: Deep Direct Reinforcement Learning for Financial Signal Representation and Trading 
[IEEE]

Network structure


(prices + last-action) - as input
Shared Block
Block1 + Block2 + Block3 + Block4    :Block1 feed the value to fill in the LSTMCell
                                     :Block2 feed the "write value" if 1 the cell will be overwrited, if 0 no values will be written over it
                                     :Block3 feed the "reset value" if 0 the cell will be resetted
                                     :Block4 if 0 noone will read the output, if 1 the output will be "public"
                                     
LSTM
DecisionBlock
Outcome
----------------------------------------------------------------------------"""

#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import sys
#------------------------------------------------------------------------------
# Dataset reading
#------------------------------------------------------------------------------

dataset_name = sys.argv[1]
testset_name = sys.argv[2]

len_ = int(sys.argv[3])
ds_ = int(sys.argv[4])

#we help recurrent neural network giving at the same time 5 last derivates
window = int(sys.argv[5])

stock_price = np.load(dataset_name)
stock_price_test = np.load(testset_name)

z = stock_price[:ds_,1:len_] - stock_price[:ds_,:len_-1]
z_test = stock_price_test[:,1:len_] - stock_price_test[:,:len_-1]
"""v = np.var(z)
mean = np.mean(z)

z = (z - mean)/ v"""
min_ = np.min(z)
max_ = np.max(z)

def norm(x):
    global min_, max_
    return (x - min_)/(max_-min_) * 2.

def denorm(x):
    global min_, max_
    return min_ + x*(max_-min_)/2.
    
z = norm(z)
z_test = norm(z_test)

n_episodes, series_length = z.shape
test_size, _ = z_test.shape
#z is a two-dim matrix, (number_episodes, series_length-1)

#------------------------------------------------------------------------------
# Model variables
#------------------------------------------------------------------------------

n_neuron = 30 #number of neuron each layer
n_layer = 5 #number of neuron of the network
#The last layer will be connected to the first one to implement the recurrent model
f = tf.nn.tanh

batch_size = 2
n_batch = n_episodes / batch_size
n_iter = 150

gamma = 1 #discount factor
c = 0.5

series_length -= window
#------------------------------------------------------------------------------
# Model definition
#------------------------------------------------------------------------------
    
def Block(input_, input_size, neuron_list,f, variables):
    last_input = input_
    last_input_size = input_size
    for neurons in neuron_list:
        std = 1./np.sqrt(input_size + 0.0)
        W = tf.Variable(tf.random_normal([last_input_size,neurons],0.,std))
        b = tf.Variable(tf.random_normal([neurons],0.,std))
        last_input = f(tf.matmul(last_input, W) + b)
        last_input_size = neurons
        variables.append(W)
        variables.append(b)
    return last_input

#(content of the cell, real output)
def Lstm(input_, write_, reset_, output_, last_lstm):
    lstm = input_*write_ + reset_*last_lstm
    return (lstm, lstm*output_)        

#this represent the reward signal
def d(u,c,  z_t, z_tm1):
    return u * z_t - c*tf.abs(z_t - z_tm1)

#input
Z_in = []
for _ in range(series_length):
    Z_in.append(tf.placeholder("float", [None,window]))
    
#input
Z = []
for _ in range(series_length):
    Z.append(tf.placeholder("float", [None,1]))
    
out = []
reward = []

print "BUILDING THE MODEL.."
    
variables=[]

old_action = 0.
lstm = np.zeros((1,10))

for t in xrange(series_length):
    sharedBlock = Block(Z_in[t], window, [10]*2, tf.tanh, variables)
    block1 = Block(sharedBlock, 10, [10]*2, tf.tanh, variables)
    block2 = Block(sharedBlock, 10, [10]*2, tf.tanh, variables)
    block3 = Block(sharedBlock, 10, [10]*2, tf.tanh, variables)
    block4 = Block(sharedBlock, 10, [10]*2, tf.tanh, variables)
    lstm, lstm_out = Lstm(block1, block2, block3, block4, lstm)
    outerBlock = Block(lstm_out, 10, [10,1], tf.tanh, variables)
    
    out_temp = outerBlock
    out.append(outerBlock)
    
    if t==0:
        reward.append(d(old_action,c,denorm(Z[t]), 0.))
    else:   
        reward.append(d(old_action,c,denorm(Z[t]), denorm(Z[t-1])))
        
    old_action =out_temp
    
r = 0.
for i in xrange(series_length):
    r = r + tf.reduce_sum(reward[i]) #* (gamma ** i)

# we should max r, or the same min -r
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-r)

print "LET's RUN THE MODEL"
# Initializing the variables

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for it in xrange(n_iter):
        
        #------------TRAIN        
        m_rew = 0
        #np.random.shuffle(z)
        for batch in xrange(n_batch):
            feed_dict = {}
            for i in range(series_length):
                feed_dict[Z_in[i]] = z[batch*(batch_size):(batch+1)*batch_size,i:i+window]
                feed_dict[Z[i]] = z[batch*batch_size:(batch+1)*batch_size,window+i-1:window +i]
            #print "expected", np.array([y_data[:]])
            rew, _= sess.run( [r, optimizer],feed_dict=feed_dict)
            n_rew = rew/ (batch_size + 0.0)
            m_rew += n_rew
            #print "predicted", m
            #print "iter",it, "batch", batch,"reward:", n_rew
        #------------TEST
        feed_dict = {}
        for i in range(series_length):
                feed_dict[Z_in[i]] = z_test[:,i:i+window]
                feed_dict[Z[i]] = z_test[:,window+i-1:window +i]
        rew = sess.run( [r],feed_dict=feed_dict)
        test_rew = rew[0] /(test_size + 0.0)
        print "["+ str(it)+  "]" + "> TRAIN: " , m_rew /(n_batch + 0.0), "TEST:", test_rew

