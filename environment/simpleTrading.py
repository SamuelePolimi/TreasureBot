import numpy as np
"""
The purpose of this class is to simulate a trading, with 3 possible positions:

- Short (0)
- Long (1)
- Neutral (2)

Thus the action space is discrete {0,1,2}. The state space has 10 dimensions:

[0] Trading signal (price)
[1] Derivate of the trading signal [0] (x_{t} - x_{t-1})
[2] Exponential Moving Average with alpha = 0.5 of [1]
[3] Exponential Moving Average with alpha = 0.25 of [1]
[4] Exponential Moving Average with alpha = 0.125 of [1]
[5] Exponential Moving Average with alpha = 0.0625 of [1]
[7] Time passed from the opening of a position (-1 if no position is open)
[8] Difference of price from the opening opening of the last position up to now. Every ticks is 0.01 (*)
[9] Actual position - 1: (-1,0,1)

Ten dimensions requires a LOT of data. If we would like theoretically to fit each dimensions with 10 different values (except for the last which needs only 3)
we need 3 x 10^9.
Dimension [7], [8], [9] could be generated artificially (i.e. every run of the program could produce arbitrarly different values).
The first dimensions that could be avoided are (imo) [0],[5].

The length of the signal should be at least longer than 100, since that the EMA [5] start to "forget" after 100 steps 
 --->    (1-0.0315)**100 = 0.04073451084527992
So the first 100 steps are devoted to "warm up" the moving averages. Then the algorithm could start to interact with the environment.
The environment is assumed to be markovian. It has discrete actions. 
Thus it could be solved with a lot of different RL methods:
Q-Learning, Sarsa, FQI, ...
The discount factor chosen is gamma=0.99.
For FQI, we even could set gamma=1, since we do a finite number of iterations. 

The negative rewards, are rescaled so that gamma=1 (otherwise the RL algorithm could potentially find a policy that is better to produce a big negative reward but in the future)
There is a real fee which is set to be 0.2%. The user could also set a virtual fee, which is only used to train the algorithm, but then all the mesurements of performance will not keep account of it.
The virtual fee is divided by
- Constant fee (every open position) [its role is to avoid positions that have only a little gain]
- Relative fee 

The environment simulate the possibility that a place order is not executed instantaneously:
For example:
PlaceBuyOrder(price):
  if actual_price <= price:
     if rnd() >= p:
        Buy()
  nextTransition()
"""

class SimpleTrading:

    def __init__(self, signal, ema_alpha=(1.,0.5,0.25,0.125,0.0625), p=0.85, fee=0.2, v_fee_const=0.01, v_fee_rel=0):
        self.signal = signal
        self.ema_alpha = np.array(ema_alpha)
        self.p = p
        self.fee = fee
        self.v_fee_const = v_fee_const
        self.v_fee_rel = v_fee_rel
        self.t = 1
        self.ema = np.ones((len(ema_alpha))) * (self.signal[1] - self.signal[0])
        self.last_position_price = 0
        self.actual_position = 0
        self.actual_position_time = 0
        self.gain = 0
        self.sell_price = 0
        self.buy_price = 0
        self.partial_gain = 0
        self.reward = 0
        self.open_sell = False
        self.open_buy = False
        for _ in xrange(0,100):
            self.ema_update()
            self.t += 1

    def step(self, action):
        self.t = self.t + 1
        self.ema_update()

        if self.open_sell:

        if self.actual_position == 1: # Long
            if action!=1:
                if not self.open_buy:
                    self.place_sell(self.signal[self.t])
        elif self.actual_position == 2: # Short
            if action!=2:
                if not self.open_sell:
                    self.place_buy(self.signal[self.t])
        if self.actual_position == 0:
            if action==1: #Long
                self.place_buy(self.signal[self.t])
                self.actual_position = 1
                self.last_position_price = self.signal[self.t]
            if action==2: #Short
                self.place_sell(self.signal[self.t])
                self.actual_position = 2
                self.last_position_price = self.signal[self.t]
        else:
            self.actual_position_time += 0.01

        r = self.reward
        self.reward = 0
        state = np.concatenate(([self.signal[self.t]],self.ema,[self.actual_position_time,self.actual_position_time, self.actual_position -1]))
        return state,r

    def ema_update(self):
        self.ema = (1 - self.ema_alpha) * self.ema + self.ema_alpha * (self.signal[self.t] - self.signal[self.t - 1])

    def place_buy(self, price):
        self.open_buy = True
        self.buy_price = price

    def try_to_buy(self):
        if self.signal[self.t] <= self.buy_price:
            if np.random.rand() <= self.p:
                self.buy()
                if self.actual_position==2:
                    self.close_position()

    def try_to_sell(self):
        if self.signal[self.t] >= self.sell_price:
            if np.random.rand() <= self.p:
                self.sell()
                if self.actual_position==1:
                    self.close_position()

    def place_sell(self, price):
        self.open_sell = True
        self.sell_price = price


    def close_position(self):
        self.actual_position_time = 0
        self.last_position_price = 0
        self.reward = self.partial_gain
        self.partial_gain = 0
        self.actual_position = 0

    def buy(self):
        self.open_buy = False
        self.gain -= self.signal[self.t]
        self.partial_gain -= self.signal[self.t]

    def sell(self):
        self.open_sell = False
        self.gain += self.signal[self.t]
        self.partial_gain += self.signal[self.t]