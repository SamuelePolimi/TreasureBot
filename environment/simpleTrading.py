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
import numpy as np

NEUTRAL = 0
LONG = 1
SHORT = 2

BUY = 0
SELL = 1


class Action:

    def __init__(self, duration, gain, type):
        self.duration = duration
        self.gain = gain
        self.type = type


class Order:

    def __init__(self, type, price, amount=1):
        self.type = type
        self.price = price
        self.amount = amount
        self.open = True

    def refresh(self, environment):
        price = environment.signal.get_price()
        logic = [lambda x,y: x<= y, lambda x,y: x>=y]
        if logic[self.type](price, self.price):
            if np.random.rand() < environment.p:
                # TODO: check if the fee formula is correct
                environment.account.budget += (self.type*2-1) * price - abs(price) * environment.fee
                self.open = False


class Position:

    def __init__(self, type, price, amount=1):
        self.type = type
        self.price = price
        self.duration = 0
        self.amount = amount
        self.order = Order(SELL if type == SHORT else BUY,price,amount)
        self.open = True
        self.closing = False

    def refresh(self, environment):
        price = environment.signal.get_price()

        if not self.order is None:
            self.order.refresh(environment)
            if not self.order.open:
                self.order = None

        if self.closing:
            if self.order is None:
                self.order = Order(BUY if self.type == SHORT else SELL, price, self.amount)
                self.order.refresh(environment)

            if self.order.type == (SELL if self.type == SHORT else BUY):
                self.order = None
                self.open = False
            else:
                if not self.order.open:
                    self.order = None
                    self.open = False


    def try_close(self):
        self.closing = True


class Account:

    def __init__(self, budget=0):
        self.budget = 0

    def update(self, value):
        self.budget += value


class Timer:

    def __init__(self):
        self.time = 0

    def update(self, value):
        self.time += value

class FinancialSignal:

    def __init__(self, price, time, frequency=10):

        self.price = price
        self.time = time
        self.frequency = frequency
        self.delta_my_time = 0
        self.delta_signal_time = time[0]
        self.index = 0
        self.end = False

    def tick(self, timers=()):

        if self.delta_my_time >= self.delta_signal_time:
            self.delta_my_time -= self.delta_signal_time
            self.index += 1
            if self.index >= len(self.price):
                self.end = True
            return False #Not another action available yet
        else:
            self.delta_my_time += self.frequency
            for timer in timers:
                timer.update(self.frequency)
            return True #Another action available

    def get_price(self):
        return self.price[self.index]


class SimpleTrading:

    def __init__(self, signal, budget=0, p=0.8, fee=0.002):
        self.signal = signal
        self.account = Account(budget)
        self.p = p
        self.fee = fee
        self.position = None

    def step(self, action):
        if not self.position is None:
            if action != self.position.type:
                self.position.try_close()
                self.position.refresh(self)
            if not self.position.open:
                self.position = None

        if self.position is None and action!=NEUTRAL:
            self.position = Position(action, self.signal.get_price())

        while not self.signal.tick():
            if self.position is not None:
                self.position.refresh(self)
                if not self.position.open:
                    self.position = None
            if self.signal.end:
                return False
        return True