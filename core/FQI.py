from sklearn.ensemble import ExtraTreesRegressor
import numpy as np

class FQI:

    def __init__(self, n_action, gamma=0.99):
        self.Q = map(lambda x: ExtraTreesRegressor(n_estimators=50),[None]*n_action)
        self.n_action = n_action
        self.gamma = gamma

    def fit(self,s,a,r,s_next):
        q_next = np.zeros_like(r)

        for a_i in range(self.n_action):
            q_next = np.concatenate((q_next,self.Q[a_i].predict(s_next)), axis=1)
        q_max = np.max(q_next, axis=1)
        for a_i in range(self.n_action):
            indx = np.argwhere(a==a_i)

            y = r[indx] + self.gamma * q_max
            self.Q[a_i].fit(s[indx,:],y)

    def take_best_action(self,s):
        q = np.zeros([0])
        for a_i in range(self.n_action):
            q = np.concatenate((q,self.Q[a_i].predict(s)))
        return np.asscalar(np.argmax(q))
