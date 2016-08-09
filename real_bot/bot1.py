"""
This Bot is executed with the structure of direct_rl.py:

It is the real robot, taking only integer action. It use the recurrent network computed with direct_rl. (just put the file outcome in here.. W_x.npy, ...)
Then give a serie in numpy format.

The results will be computed with just a feed forward used like a recurrent (so that it can be used for an unfixed length of time)

ASAP I'll provide a chart showing when the program is buying and when is selling through the time_serie :)
"""

import numpy as np
import tensorflow as tf
import sys 

W_x_real = np.load("W_x.npy")
B_x_real = np.load("B_x.npy")
W_m_real = np.load("W_m.npy")
B_out_real = np.load("B_out.npy")
W_out_real = np.load("W_out.npy")
min_, max_ = tuple(np.load("min_max.npy"))

n_neuron = W_x_real.shape[1]
window = W_x_real.shape[0]

def norm(x):
    global min_, max_
    return (x - min_)/(max_-min_) * 2.

def denorm(x):
    global min_, max_
    return min_ + x*(max_-min_)/2.
    
serie = np.load(sys.argv[1])
z = serie[:,1:] - serie[:,:-1]
z = norm(z)
 
cost = float(sys.argv[2])
#------------------------------------------------------------------------------
# Definition of the model
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

#this is the action to perform    
def u(X, W_x, B_x):
    #tanh is bounded between 1 and -1
    return tf.tanh(tf.matmul(X, W_x) + B_x)

#this represent the reward signal
def d(u,c,  z_t, z_tm1):
    return u * z_t - c*tf.abs(z_t - z_tm1)

W_x = map(tf.constant, W_x_real)
B_x = map(tf.constant, B_x_real)
W_m = tf.constant(W_m_real)
W_out = tf.constant(W_out_real)
B_out = tf.constant(B_out_real)

#input
Z = tf.placeholder("float", [None,window])
#TODO: 30 should be pick'd up from the size of W_x for example
M = tf.placeholder("float", [1,10])  


print "BUILDING THE MODEL.."
f = tf.tanh

h = add_first_layer(Z,W_x[0], B_x[0],M, W_m , f)   
for i in xrange(1,len(W_x)):
    h = add_layer(h, W_x[i], B_x[i],f)
out = u(h,W_out,B_out)


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    m = np.zeros((1,10))
    a = 0
    a_old=0
    rew = 0
    for i in xrange(z.shape[1] - window):
        feed_dict = {}
        feed_dict[Z] = z[0:1,i:i+window]
        feed_dict[M] = m
    
        m, o = sess.run( [h,out],feed_dict=feed_dict)
    
        a = 0
        if o> 1./3.:
            a = 1
        if o< -1./3.:
            a = -1
            
        rew += a_old * denorm(z[0,window+i-1])
        if a != a_old:
            rew -= cost
        a_old = a
        print "Action: ", a, "reward:", rew