"""----------------------------------------------------------------------------
This regressor optimizes the gain with long short action.
You should provide to the regressor a trainset and a validationset with the following shape:

(nrow, series_length, n_series*2 + n_features)

remember that in position

[i,j,0:n_series]

we will find the actual prices at time j of the stocks

[i,j,n_series:n_series*2]

we will find the atual cost of performing a long or a short for the stocks at the time j

at position: [i,j,n_series*2:]

we will find the l-th features of the stocks at the time j.

The shapes of this regressor is described:

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
from optimizer import Optimizer
import numpy as np
import tensorflow as tf

def Block(input_, input_size, neuron_list,f, variables, dropout=1.):
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
    
"""
u_tm1 : action at time t-1
u_t : action at time t
z_t : value of derivative at time t
c : cost
"""
def d(u_tm1, u_t, z_t, c):
    #u_tm1 * z_t : gain at time t (you use action in precedence, because the agent decide now based on what saw before)
    #c * |u_t - u_tm1| is the cost of change action
    #divided by 2 because of a correction
    return u_tm1 * z_t - c * tf.abs(u_t-u_tm1) / 2.
    
class LSTMNet(Optimizer):
    
    #dimension of blocks are expressed by (n_layer, n_neuron)
    def __init__(self, optimizerConfig, train_set, validation_set, n_series, features):
        
        self.sharedBoxShape = optimizerConfig['sharedBoxShape']
        self.blocksShape = optimizerConfig['blocksShape']
        self.nLSTMCells = optimizerConfig['nLSTMCells']
        self.decisionBlockShape = optimizerConfig['decisionBlockShape']
        
        self.train_set = train_set
        self.validation_set = validation_set
        
        assert len(train_set.shape) == 3 , "train_set should be 3 dimensional"
        assert len(validation_set.shape) == 3, "validation_set should be 3 dimensional"
        assert train_set.shape[1] == validation_set.shape[1], "validation and train sets should have 2nd dimension equal (serie_length)"
        assert train_set.shape[2] == validation_set.shape[2], "validation and train sets should have 3rd dimension equal (n_serie*2 + n_features)"
        assert train_set.shape[2] >= 2
        
        self.serie_length = self.train_set.shape[1]
        assert n_series >= 1, "Provide at least one serie" 
        assert features + n_series * 2 == self.train_set.shape[2], "the third dimension of the datasets should be equal to number of features + n_series * 2"
        self.n_series = n_series
        
        self.features = features
        self.train_set = train_set
        self.validation_set = validation_set
        self.dropout = optimizerConfig['dropout']
        self.batch_size=optimizerConfig['batch_size']
        #here I find the mean of each feature
        mean = np.mean(np.mean(train_set, axis=0),axis=0)
        
        #here I find the mean of each feature
        std = np.std(np.std(train_set, axis=0),axis=0)
        
        
        """Just a right normalization (between 0 and 1)
        """
        self.norm_prices = lambda x: (x - mean[0:n_series]) / std[0:n_series]
        self.denorm_prices = lambda x: x * std[0:n_series] + mean[0:n_series]
        
        self.norm_costs = lambda x: (x - mean[n_series:n_series*2]) / std[n_series:n_series*2]
        self.denorm_costs = lambda x: x * std[n_series:n_series*2] + mean[n_series:n_series*2]
        
        self.norm_features = lambda x: (x - mean[n_series*2:]) / std[n_series*2:]
        self.denorm_features = lambda x: x * std[n_series*2:] + mean[n_series*2:]
        
        self.initialize()
        
    def initialize(self):
        
        n_series = self.n_series
        features = self.features
        serie_length = self.serie_length
        variables=[]
        
        # the last action performed (default is 0 - Neutral)
        old_action = 0.
        # this is the content of the lstm cells at time -1
        lstm = np.zeros((1,self.nLSTMCells)).astype(np.float32)
        # this is the output of the lstm at time -1
        lstm_out = np.zeros((1,self.nLSTMCells)).astype(np.float32)
        
        # output of the network (decision) through time
        out = []
        # reward through time
        reward = []
        
        #Stock price derivative
        Z = []
        for _ in range(serie_length):
            Z.append(tf.placeholder("float", [None,n_series]))
        
        #Stock cost
        C = []
        for _ in range(serie_length):
            C.append(tf.placeholder("float", [None,n_series]))
                    
        #Features
        F = []
        for _ in range(serie_length):
            F.append(tf.placeholder("float", [None,features]))
        
        # unfold through time
        for t in xrange(serie_length):
            # As we remember the shape of the dataset is (n_row, length, n_series, n_features)
            # each Z should be a vertical vector (n_row, 1, n_series, n_features)
            
            # Merge of the input
            
            print "Unfold: ", t+1, "out of", serie_length

            inputShared1 = Merge([self.norm_prices(Z[t]),self.norm_costs(C[t]),self.norm_features(F[t])],[n_series,n_series,features],n_series*2 + features,tf.tanh,variables)
          
            # Shared block 1: elaboration of the input
            sharedBlock1 = Block(inputShared1, n_series*2 + features , [self.sharedBoxShape[1]]*self.sharedBoxShape[0], tf.tanh, variables, dropout=self.dropout)
            
            # Features given by shared1 and lstm
            inputShared2 = Merge([sharedBlock1, lstm_out]
                ,[self.sharedBoxShape[1],self.nLSTMCells]
                , self.sharedBoxShape[1] ,tf.tanh, variables)   


            sharedBlock2 = Block(inputShared2, self.sharedBoxShape[1], [self.sharedBoxShape[1]] * self.sharedBoxShape[0], tf.tanh, variables, dropout=self.dropout)
            
            # Each block represent a gate for the LSTM Cells
            block1 = Block(sharedBlock2, self.sharedBoxShape[1], [self.blocksShape[1]] * self.blocksShape[0] + [self.nLSTMCells], tf.tanh, variables, dropout=self.dropout)
            block2 = Block(sharedBlock2, self.sharedBoxShape[1], [self.blocksShape[1]] * self.blocksShape[0] + [self.nLSTMCells], tf.tanh, variables, dropout=self.dropout)
            block3 = Block(sharedBlock2, self.sharedBoxShape[1], [self.blocksShape[1]] * self.blocksShape[0] + [self.nLSTMCells], tf.tanh, variables, dropout=self.dropout)
            block4 = Block(sharedBlock2, self.sharedBoxShape[1], [self.blocksShape[1]] * self.blocksShape[0] + [self.nLSTMCells], tf.tanh, variables, dropout=self.dropout)
            
            #LSTM cells
            lstm, lstm_out = Lstm(block1, block2, block3, block4, lstm)
            outerBlock = Block(lstm_out, self.nLSTMCells, [self.decisionBlockShape[1]] * self.decisionBlockShape[0] + [n_series], tf.tanh, variables, dropout=self.dropout)
            
            out_temp = outerBlock
            out.append(outerBlock)
        
            reward.append(tf.reduce_sum(d(old_action,out_temp, self.denorm_prices(Z[t]), self.denorm_costs(C[t]))))
            
            old_action = out_temp
    
        r = 0.
        for i in xrange(serie_length):
            r = r + tf.reduce_sum(reward[i])

        # we should max r, or the same min -r
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-r)
        self.tot_reward = r
        self.out = out
        
        self.Z = Z
        self.C = C
        self.F = F
        self.variables = variables
        
        init = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(init)
        
        
    def learn(self):
        
        m_rew = 0
        if True:#shuffle:
            np.random.shuffle(self.train_set)
        
        batch_size = self.batch_size
        n_batch = self.train_set.shape[0] / batch_size
        n_series = self.n_series
        Z , C , F = (self.Z, self.C, self.F)
        
        #-----------TRAIN
        for batch in xrange(n_batch):
            feed_dict = {}
            for i in range(self.serie_length):
                feed_dict[Z[i]] = self.train_set[batch*batch_size:(batch+1)*batch_size,i,0:n_series]
                feed_dict[C[i]] = self.train_set[batch*batch_size:(batch+1)*batch_size,i,n_series:n_series*2]
                feed_dict[F[i]] = self.train_set[batch*batch_size:(batch+1)*batch_size,i,n_series*2:]
                
            rew, _= self.session.run( [self.tot_reward, self.optimizer],feed_dict=feed_dict)
            n_rew = rew / (batch_size + 0.0)
            m_rew += n_rew
            
        #------------TEST
        feed_dict = {}
        for i in range(self.serie_length):
            feed_dict[Z[i]] = self.validation_set[:,i,0:n_series]
            feed_dict[C[i]] = self.validation_set[:,i,n_series:n_series*2]
            feed_dict[F[i]] = self.validation_set[:,i,n_series*2:]
        rew = self.session.run( [self.tot_reward],feed_dict=feed_dict)
        test_rew = rew[0] /(self.validation_set.shape[0] + 0.0)
        
        return (m_rew /(n_batch + 0.0), test_rew)
        
    def finalize(self):
        raise("Not implemented yet")
        return self.session.run(self.variables)
        