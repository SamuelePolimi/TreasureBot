from environment import SimpleTrading
from environment.simpleTrading import FinancialSignal
import numpy as np
import matplotlib.pyplot as plt

market = "BTCEUR"

signal = np.load("../dataset/" + market +  "/train.npy")
print signal
env = SimpleTrading(FinancialSignal(signal.item().get("price"),signal.item().get('time'),frequency=100.))

my_signal = []
my_wallet = []

for _ in xrange(50):
    my_signal.append(env.signal.get_price())
    env.signal.tick()

def plot(num, title, s,legend=None, y=None, clf=True, color='b'):
    plt.figure(num)
    if clf:
        plt.clf()
    plt.title(title)
    plt.legend(loc='best')
    if y is not None:
        plt.plot(y,s, color=color)
    else:
        plt.plot(range(len(s)),s,color=color)

plot(1,"price",my_signal)
plt.pause(0.0001)
cont = True
N_window = 100
active_time = 0
while cont:
    print "Wallet: ", env.account.budget
    x = raw_input("NEUTRAL 0, LONG 1, SHORT 2: ")
    info = env.step(int(x))
    print info
    cont = info['end']
    if info['active']:
        active_time += 1
    if info['position']==0:
        active_time =0
    N_window = max([100,active_time])
    my_signal.append(env.signal.get_price())
    my_signal = my_signal[-N_window:]
    my_wallet.append(env.account.budget)
    my_wallet = my_wallet[-N_window:]
    plot(1,"price",my_signal)
    if active_time != 0 :
        plot(1,"price",
             [info['open_price'], env.signal.get_price()],
             y=[len(my_signal) - active_time-1,len(my_signal)-1]
             ,clf=False,
             color='r' if info['gain'] < 0 else 'g' )
    plot(2,"wallet",my_wallet)
    plt.pause(0.0001)



