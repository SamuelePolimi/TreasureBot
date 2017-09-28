import numpy as np
import matplotlib.pyplot as plt
from environment import SimpleTrading

signal = np.linspace(0,2,202)
actions = [0]*10 + [1]*20 + [0]*10 + [2] * 20 + [1]*20 + [0]*20

env = SimpleTrading(signal)
for a in actions:
    print env.t, env.step(a)[1]
