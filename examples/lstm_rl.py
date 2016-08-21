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
# Parameter passing
#------------------------------------------------------------------------------

def retreiveParams():
    global folder,test,case,code, number, library
    commands = sys.argv[1:]
        
    for arg in commands:
        s = arg.split("=")    
        name = s[0]
        value = s[1]
        try:
            if "." in value:
                globals()[name] = float(value)
            else:
                globals()[name] = int(value)
        except:
            globals()[name] = str(value)

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------

"""----------------------------------------------------------------------------
" All of these parameter are callable from the command line in the shape:
" python example/lstm_rl.py <variable_name>=<value>
" for example:
" python example/lstm_rl.py shuffle=1 lstm_neuron=10 c=0.0013 dropout=0.8
"---------------------------------------------------------------------------"""

block_layer = 3  #number of neuron each layer
block_neuron = 10
base_neuron = 10
base_layer = 1
lstm_neuron = 10
outcome_layer = 2
outcome_neuron = 10
dropout=1.
shuffle=False

#The last layer will be connected to the first one to implement the recurrent model
f = tf.nn.tanh

batch_size = 10
n_iter = 150

gamma = 1 #discount factor
c = 0.0019

dataset_folder = "apple"

#length of the time serie
len_ = 0
#how many data can I pick up
ds_ = 0

#we help recurrent neural network giving at the same time 5 last derivates
window = 1

retreiveParams()

#------------------------------------------------------------------------------
# Data reading & preformatting
#------------------------------------------------------------------------------

try:
    dataset_name = "dataset/" + dataset_folder + "/train.npy"
    testset_name = "dataset/" + dataset_folder + "/test.npy"
    
    stock_price = np.load(dataset_name)
    stock_price_test = np.load(testset_name)
    
    com = c * stock_price[:ds_,:len_]
    com_test = c * stock_price_test[:,:len_]
    
    z = stock_price[:ds_,1:len_] - stock_price[:ds_,:len_-1]
    
    print z.shape
    z_test = stock_price_test[:,1:len_] - stock_price_test[:,:len_-1]
    """v = np.var(z)
    mean = np.mean(z)
    
    z = (z - mean)/ v"""
    min_ = np.min(z)
    max_ = np.max(z)
    
    com_min_max = (np.min(com), np.max(com))
    z_min_max = (min_,max_)
    
    def norm(x,min_max):
        (min_,max_) = min_max
        return (x - min_)/(max_-min_) * 2.
    
    def denorm(x,min_max):
        (min_,max_) = min_max
        return min_ + x*(max_-min_)/2.
        
    z = norm(z, z_min_max)
    z_test = norm(z_test, z_min_max)
    
    com = norm(com, com_min_max)
    com_test = norm(com_test, com_min_max)
    
    n_episodes, series_length = z.shape
    test_size, _ = z_test.shape
    
    
    n_batch = n_episodes / batch_size
    series_length -= window
except:
    print "Error in section DATA READING & PREFORMATTING ----------------------"
    print "\tThis is probably caused by an error on the parameter passing."
    print "--------------------------------------------------------------------"
    print "\tTo run this program is sufficient to type the command in the following shape:"
    print "\tpython example/lstm_rl.py <parameter_name>=<value> ..."
    print "\tEG:\n"
    print "\t\tpython example/lstm_rl.py base_layer=2 shuffle=1 c=0.0018\n"
    print "\tThe parameter names are the same of the fariables declared in the section parameters"
    print "\tYou don't need to type all of them - the ones you don't type will be set on the default value"
    print "\tYou also will need two dataset files (.npy) in the folder dataset/<dataset_folder>"
    print "\tThe files in <dataset_folder> (eg: dataset/apple) should be named 'train.npy' & 'test.npy'"
    print "\tActually in future version of this program, we will better have 'train.npy', 'validation.npy' and 'test.npy'"
    print "\tHAVE FUN! :)"
    print "--------------------------------------------------------------------"
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
        last_input = tf.nn.dropout(last_input,dropout)
        last_input_size = neurons
        variables.append(W)
        variables.append(b)
    return last_input

#(content of the cell, real output)
def Lstm(input_, write_, reset_, output_, last_lstm):
    lstm = input_*write_ + reset_*last_lstm
    return (lstm, lstm*output_)        

def Merge(input_list, dim_list, out_dim, f,variables):
    sum_ = np.zeros((1,out_dim))
    for input_, dim_ in zip(input_list, dim_list):
        std = 1./np.sqrt(dim_ + 0.0)
        W = tf.Variable(tf.random_normal([dim_,out_dim],0.,std))
        sum_ = sum_ + tf.matmul(input_,W)
        variables.append(W)
    b = tf.Variable(tf.random_normal([out_dim],0.,std))
    variables.append(b)
    return f(sum_+b)
    
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

Comm = []
for _ in range(series_length):
    Comm.append(tf.placeholder("float", [None,1]))
    
out = []
reward = []

print "BUILDING THE MODEL.."
    
variables=[]

old_action = 0.
lstm = np.zeros((1,10))
lstm_out = np.zeros((1,10))

for t in xrange(series_length):
    inputShared1 = Merge([Z_in[t],Comm[t]],[window, 1],10,tf.tanh,variables)
    sharedBlock1 = Block(inputShared1, 10, [10]*2, tf.tanh, variables)
    inputShared2 = Merge([sharedBlock1, lstm_out],[10],10,tf.tanh, variables)    
    sharedBlock2 = Block(inputShared2, 10, [10], tf.tanh, variables)
    block1 = Block(sharedBlock2, 10, [10]*3, tf.tanh, variables)
    block2 = Block(sharedBlock2, 10, [10]*3, tf.tanh, variables)
    block3 = Block(sharedBlock2, 10, [10]*3, tf.tanh, variables)
    block4 = Block(sharedBlock2, 10, [10]*3, tf.tanh, variables)
    lstm, lstm_out = Lstm(block1, block2, block3, block4, lstm)
    outerBlock = Block(lstm_out, 10, [10,10,1], tf.tanh, variables)
    
    out_temp = outerBlock
    out.append(outerBlock)
    
    #the commission is halfed because
    #going from 0 to 1 you pay one
    #going from 1 to 0 you pay one (but you should pay 0 because you already paid)
    #going from 0 to -1 you pay 1
    #going from -1 to 0 you pay 1 (and you should pay 0)
    #going from -1 to 1 you pay 2 (and you should pay 1)
    #goind from 1 to -1 you pay 2 (and you should pay 1)
    #so the average cost is 8./6. but it should be 4./6. (assuming each kind of transaction happens with same probability). So 8./6. * 1./2. = 4./6.
    
    if t==0:
        reward.append(d(old_action,denorm(.5 * Comm[t], com_min_max),denorm(Z[t], z_min_max), 0.))
    else:   
        reward.append(d(old_action,denorm(.5 * Comm[t], com_min_max),denorm(Z[t], z_min_max), denorm(Z[t-1], z_min_max)))
        
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
        if shuffle:
            np.random.shuffle(z)
        for batch in xrange(n_batch):
            feed_dict = {}
            for i in range(series_length):
                feed_dict[Z_in[i]] = z[batch*(batch_size):(batch+1)*batch_size,i:i+window]
                feed_dict[Z[i]] = z[batch*batch_size:(batch+1)*batch_size,window+i-1:window +i]
                feed_dict[Comm[i]] = com[batch*batch_size:(batch+1)*batch_size,i:i+1]
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
                feed_dict[Comm[i]] = com_test[:,i:i+1]
        rew = sess.run( [r],feed_dict=feed_dict)
        test_rew = rew[0] /(test_size + 0.0)
        print "["+ str(it)+  "]" + "> TRAIN: " , m_rew /(n_batch + 0.0), "TEST:", test_rew

