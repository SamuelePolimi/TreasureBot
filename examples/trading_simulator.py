from environment import SimpleTrading
from environment.simpleTrading import FinancialSignal
import numpy as np
import matplotlib.pyplot as plt

market = "BTCEUR"
signal = np.load("../dataset/" + market + "/test.npy")
print signal
env = SimpleTrading(FinancialSignal(signal.item().get("price"),signal.item().get('time'),frequency=1.))

my_signal = []
my_wallet = []

for _ in xrange(50):
    my_signal.append(env.signal.get_price())
    env.signal.tick()

def plot(num, title, s):
    plt.figure(num)
    plt.clf()
    plt.title(title)
    plt.plot(s)

plot(1,"price",my_signal)
plt.pause(0.0001)
cont = True
while cont:
    print "Wallet: ", env.account.budget
    x = raw_input("NEUTRAL 0, LONG 1, SHORT 2: ")
    cont = env.step(int(x))
    my_signal.append(env.signal.get_price())
    my_signal = my_signal[-100:]
    my_wallet.append(env.account.budget)
    my_wallet = my_wallet[-100:]
    plot(1,"price",my_signal)
    plot(2,"wallet",my_wallet)
    plt.pause(0.0001)



