"""----------------------------------------------------------------------------

Created on Fri Jul 29 15:53:23 2016

@author: samuele

This is just a rough example on how to use Direct Reinforcement Learning in Stock Exchange (only one stock).

Our model is a RNN which the output is 

d_t = u_t * z_t - c * (u_t- u_(t-1))

where u_t is bounded between -1 and 1 and means buy and sell.
z_t is the derivate of the stock price (in practice p_t - p_(t-1))

the second term of the formula above c * (u_t - u_(t-1)) represent the cost of buy a stock or sell it
(otherwise the bot could just buy as the price arize and sell as the price decrease) 

the model should just max the summation of d_t (that will be discounted through time by gamma)


This example will be not very efficient because of the problem of the vanishing-gradient. 
We should use LSTM, truncated Gradient descent, ... and many other technique..


Reference: Deep Direct Reinforcement Learning for Financial Signal Representation and Trading 
[IEEE]
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

stock_price = np.load(dataset_name)
stock_price_test = np.load(testset_name)

z = stock_price[:ds_,1:len_] - stock_price[:ds_,:len_-1]
z_test = stock_price_test[:,1:len_] - stock_price_test[:,:len_-1]
"""v = np.var(z)
mean = np.mean(z)

z = (z - mean)/ v"""
min_ = np.min(z)
max_ = np.max(z)


z = (z - min_)/(max_-min_) * 2.
z_test = (z_test - min_)/(max_-min_) * 2.

n_episodes, series_length = z.shape
test_size, _ = z_test.shape
#z is a two-dim matrix, (number_episodes, series_length-1)

#------------------------------------------------------------------------------
# Model variables
#------------------------------------------------------------------------------

n_neuron = 10 #number of neuron each layer
n_layer = 5 #number of neuron of the network
#The last layer will be connected to the first one to implement the recurrent model
f = tf.nn.tanh

batch_size = 100
n_batch = n_episodes / batch_size
n_iter = 1000

gamma = 1 #discount factor
c = 0.1

#------------------------------------------------------------------------------
# Model definition
#------------------------------------------------------------------------------

#the normal layer
def add_layer(X, W_x, B_x, f):
    return f( tf.matmul(X, W_x) + B_x)

#The layer with the memory from the past
def add_first_layer(X, W_x, B_x, M, W_m, f):
    temp = tf.matmul(X, W_x)
    temp += tf.matmul(M, W_m)
    temp += B_x
    return f( tf.matmul(X, W_x) + tf.matmul(M, W_m) + B_x)

def add_base_layer(X, W_x, B_x, f):
    return f( tf.matmul(X, W_x) + B_x)

#this is the action to perform    
def u(X, W_x, B_x):
    #tanh is bounded between 1 and -1
    return tf.tanh(tf.matmul(X, W_x) + B_x)

#this represent the reward signal
def d(u,c,  z_t, z_tm1):
    return u * z_t - c*(z_t - z_tm1)
    
std_ = 3.
m_=0.
#weight and biases of the network
W_x = [tf.Variable(tf.random_normal([1,n_neuron],m_,std_))]
for _ in range(1,n_layer):
    W_x.append(tf.Variable(tf.random_normal([n_neuron,n_neuron],m_,std_)))
B_x = [ tf.Variable(tf.random_normal([n_neuron],m_,std_))]
for _ in range(1,n_layer):
    B_x.append(tf.Variable(tf.random_normal([n_neuron],m_,std_)))
W_m = tf.Variable(tf.random_normal([n_neuron,n_neuron],m_,std_))
W_out = tf.Variable(tf.random_normal([n_neuron,1],m_,std_))
B_out = tf.Variable(tf.random_normal([1],m_,std_))

#input
Z = []
for _ in range(series_length):
    Z.append(tf.placeholder("float", [None,1]))
    

out = []
reward = []

print "BUILDING THE MODEL.."
    
h = add_base_layer(Z[0],W_x[0], B_x[0], f)   
for i in xrange(1,n_layer):
    h = add_layer(h, W_x[i], B_x[i],f)
out_temp = u(h,W_out,B_out)
out.append(out_temp)
reward.append(tf.reduce_sum(d(out_temp, c, Z[0], 0)))


for j in xrange(1,series_length):
    h = add_first_layer(Z[j],W_x[0], B_x[0],h,W_m ,f)   
    for i in xrange(1,n_layer):
        h = add_layer(h, W_x[i], B_x[i],f)
    out_temp = u(h,W_out,B_out)
    out.append(out_temp)
    reward.append(tf.reduce_sum(d(out_temp, c, Z[j], Z[j-1])))

r = 0 
for i in xrange(series_length):
    r = r + reward[i] * (gamma ** i)

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
        np.random.shuffle(z)
        for batch in xrange(n_batch):
            feed_dict = {}
            for i in range(series_length):
                feed_dict[Z[i]] = z[batch*batch_size:(batch+1)*batch_size,i:i+1]
            #print "expected", np.array([y_data[:]])
            rew, _= sess.run( [r, optimizer],feed_dict=feed_dict)
            n_rew = rew/ (batch_size + 0.0)
            m_rew += min_  + n_rew * (max_-min_) 
            #print "predicted", m
            #print "iter",it, "batch", batch,"reward:", n_rew
        #------------TEST
        feed_dict = {}
        for i in range(series_length):
            feed_dict[Z[i]] = z_test[:,i:i+1]
        rew = sess.run( [r],feed_dict=feed_dict)
        test_rew = rew[0] /(test_size + 0.0)
        test_rew = min_  + test_rew * (max_-min_) 
        print "["+ str(it)+  "]" + "> TRAIN: " , m_rew /(n_batch + 0.0), "TEST:", test_rew
