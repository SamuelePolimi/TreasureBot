from sklearn.ensemble import ExtraTreesRegressor
import numpy as np

class FQI:

    def __init__(self, n_action, gamma=0.99):
        self.Q = map(lambda x: ExtraTreesRegressor(n_estimators=50),[None]*n_action)
        self.n_action = n_action
        self.gamma = gamma
        self.first_time = True

    def fit(self,s,a,r,s_next):

        if not self.first_time:
            q_next = self.Q[0].predict(s_next)
            q_next = q_next.reshape((q_next.shape[0],1))
            for a_i in range(1,self.n_action):
                q_next = np.concatenate((q_next,self.Q[a_i].predict(s_next).reshape((q_next.shape[0],1))), axis=1)
            q_max = np.max(q_next, axis=1)
            for a_i in range(self.n_action):
                indx = np.argwhere(a==a_i)

                y = r[indx].ravel()+ 1 * q_max[indx].ravel() #+ self.gamma * q_max[indx].ravel()
                self.Q[a_i].fit(s[indx.ravel(),:],y)
        else:
            for a_i in range(self.n_action):
                indx = np.argwhere(a == a_i)
                y = r[indx]
                self.Q[a_i].fit(s[indx.ravel(), :], y.ravel())
            self.first_time = False

    def take_best_action(self,s):
        q = self.Q[0].predict([s])
        for a_i in range(1,self.n_action):
            q = np.concatenate((q,self.Q[a_i].predict([s])))
        return np.asscalar(np.argmax(q))
