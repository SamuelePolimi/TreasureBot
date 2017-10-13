from core import FQI
from environment import EnvironmentWrapper,SimpleTrading,FinancialSignal
import numpy as np
import matplotlib.pyplot as plt

freq = 100.
marketname = 'BTCEUR'
alphas = (1.,0.5,0.25,0.125)
signal_train = np.load('../dataset/' + marketname + "/train.npy")
signal_test = np.load('../dataset/' + marketname + "/test.npy")
signal = FinancialSignal(signal_train.item().get('price'),signal_train.item().get('time'),frequency=freq)
env_core = SimpleTrading(signal )
env_train = EnvironmentWrapper(env_core, warmup=20000,relativeSetting=False,include_price=True)
last_action = 0
ds_size = 30000
warmUpRange = (100, 20000)

test_plt = []
train_plt = []
################################
# Dataset Generation
################################
s = None
a = None
r = None
s_next = None
N_step = 1000
last_action = np.random.randint(0,3)
state = env_train.reset()
tot_reward = 0
for _ in range(N_step):
    if np.random > 0.8: #Let's take the same action 80% of times
        last_action = np.random.randint(0,3)
        next_state, reward = env_train.step(last_action)
    else:
        next_state, reward = env_train.step(last_action)
    if s is None:
        s = np.array([state])
        s_next = np.array([next_state])
        r = np.array([reward])
        a = np.array([last_action])
    else:

        s = np.concatenate((s,[state]),axis=0)
        s_next = np.concatenate((s_next,[next_state]),axis=0)
        r = np.concatenate((r,[reward]))
        a = np.concatenate((a,[last_action]))
    state = next_state
    tot_reward += reward
print tot_reward
fqi = FQI(3)

epsilon = 1.
epsilon_decay = 0.9
for i in range(300):
    print "FQI", i
    epsilon *= epsilon_decay
    fqi.fit(s,a,r,s_next)
    signal = FinancialSignal(signal_train.item().get('price'), signal_train.item().get('time'), frequency=freq)
    env = SimpleTrading(signal)
    env_test_train = EnvironmentWrapper(env, warmup=np.random.randint(warmUpRange[0],warmUpRange[1]),relativeSetting=False,include_price=True)
    state = env_test_train.reset()
    tot_reward = 0
    actions = [0,0,0]
    for _ in range(N_step):
        if np.random.rand() < epsilon:
            action = np.random.randint(0,3)
        else:
            action = fqi.take_best_action(state)
        actions[action] += 1
        next_state, reward = env_test_train.step(action)
        s = np.concatenate((s, [state]), axis=0)
        s_next = np.concatenate((s_next, [next_state]), axis=0)
        r = np.concatenate((r, [reward]))
        a = np.concatenate((a, [action]))
        state = next_state
        tot_reward += reward
        if env.position is None:
            record_reward = tot_reward
            final_budget = env.account.budget

    # indx = range(s.shape[0])
    # np.random.shuffle(indx)        if np.random.rand() < epsilon:
            action = np.random.randint(0,3)
        else:
            action = fqi.take_best_action(state)
    # s = s[indx[:ds_size],:]
    # s_next = s_next[indx[:ds_size],:]
    # a = a[indx[:ds_size]]
    # r = r[indx[:ds_size]]
    print actions
    print "train", record_reward, final_budget
    train_plt.append(record_reward)
    signal = FinancialSignal(signal_test.item().get('price'), signal_test.item().get('time'), frequency=freq)
    env = SimpleTrading(signal)
    env_test_test = EnvironmentWrapper(env, warmup=np.random.randint(warmUpRange[0],warmUpRange[1]),relativeSetting=False,include_price=True)
    state = env_test_test.reset()
    tot_reward = 0
    actions = [0, 0, 0]
    for _ in range(N_step):
        if np.random.rand() < epsilon:
            action = np.random.randint(0,3)
        else:
            action = fqi.take_best_action(state)
        actions[action] += 1
        state, reward = env_test_test.step(action)
        tot_reward += reward
        if env.position is None:
            final_budget = env.account.budget
            record_reward = tot_reward

    print actions
    print "test", record_reward, final_budget
    test_plt.append(record_reward)

    plt.clf()
    plt.plot(test_plt,label="test")
    plt.plot(train_plt, label="train")
    plt.legend(loc='best')
    plt.pause(0.001)


