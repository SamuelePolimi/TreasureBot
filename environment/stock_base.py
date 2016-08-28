import numpy as np
class BaseStock:
        
    def savePriceList(self,length, filename):
        prices = []
        n = 5000 + np.random.randint(10000)
        for _ in range(n):
            self.step(0)
        for _ in range(length):
            prices.append(self.price)
            self.step(0)
        np.save(filename,np.array(prices))
        self.reset()
        
    def saveDataset(self,n_episodes,length, filename):
        prices_matrix = []
        for _ in range(n_episodes):
            prices = []
            n = 5000 + np.random.randint(10000)
            for _ in range(n):
                self.step(0)
            for _ in range(length):
                prices.append(self.price)
                self.step(0)
            prices_matrix.append(prices)
        np.save(filename,np.array(prices_matrix))
        self.reset()
