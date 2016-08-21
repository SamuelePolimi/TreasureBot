"""
This stock simulation, is just the simulation of one of the easiest function:

Sinusoid!

The stock price here is deterministic, and it is very easy to understand when to SHORT and where to LONG.
This stock price will be used just for testing.
"""

import numpy as np
from stock_base import BaseStock

class Sin(BaseStock):
    
    """
    f = frequency of the sinusoid
    h = heigth of the sin
    c = displacement over the x axe (remember c - 0.5 h > 0)
    d = sin offset
    """
    def __init__(self, f = 0.01, h = 2., c = 10., d = 1.):
        if c - 0.5 * h < 0:
            raise "c and h leads to negative prices: this is not allowed"
        self.f = f
        self.h = h
        self.c = c
        self.d = d
        self.t = 0  #time of first step
        self.price = self.h * np.sin(self.d + self.f * 0.) + self.c
        
    def step(self, action):
        self.t += 1.
        #the following if, don't allow t to grow up to infinite
        #[1]   sin(f*t) = sin(f*t - 2pi)
        #   f*tnew = f*t - 2pi
        #   tnew = t - 2pi/f
        #   sin(f * tnew) = sin(f * (t - 2pi/f)) = sin(f*t - 2pi) = sin(f*t) (thanks to [1])
        #   
        #   sin(f * tnew) = sin(f * t)           so the code below is correct :)
        if self.t  * self.f > 2 * np.pi:
            self.t = self.t - (2. * np.pi / self.f) 
            
        self.price = self.h * np.sin(self.d + self.f * self.t) + self.c
        return action * self.price
    
    def getState(self):
        return [self.price]
        
    def getEnvironmentInformation(self):
        #dimension of state, dimension of action, dimension of reward
        return (1,1,1)
    
    def reset(self):
        self.price = self.h * np.sin(self.d + self.f * 0.) + self.c
        self.t = 0
