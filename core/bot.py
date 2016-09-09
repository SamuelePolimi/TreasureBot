

class Bot:
    
    """
    suggester = optimizer.finalize()
    n_Stock = number of stock we are buying/selling
    features = (thereIsMin_Cose, theseIsMax_cost, thereIsVolume, thereIsTiming ) Eg: (False, False, True, False): only one feature Volume
    isOneToOne: the output of the suggester is only one, so we will apply only one stock and one feature n times to the suggester. Else we will use only one suggester to decide all n actions
    capital = initial amount of money
    rate_capital = the amount that we want to invest each time from the capital (with rate_capita==0.1 we will invest 10% of the capital) 
    """
    def __init__(self, suggester, n_Stock, features, isOneToOne, capital, rate_capital):
        self.suggester = suggester
        self.n_Stock = n_Stock
        self.features = features 
        self.isOneToOne = isOneToOne
        self.resetCapital = capital
        self.capital = capital
        self.rate_capital = rate_capital
        self.virtual = [0] * n_Stock
        self.position = [0] * n_Stock
        
        self.actionHistory = []
        self.capitalHistory = []
        self.virtualCapitalHistory = []


    def discretize(a):
        if a > 1./3.:
            return 1
        elif a > -1/3.:
            return 0
        else:
            return -1
            
    def actionize(self,action):
        for i in range(0, self.n_Stock):
            if self.position[i] == 1 and action[i] != 1:
                self.capital += self.virtual[i]
                self.virtual[i] = 0
            if self.position[i] == -1 and action[i] != -1:
                self.capital += self.virtual[i]
                self.virtual[i] = 0
        self.position[i] = action[i]
        
    """
    WE have to well define this class
    """
    def step(self, state):
        
        for i in range(0, self.n_Stock):
            self.virtual[i] = self.position[i] * state[i] * self.capital * self.rate_capital 
        
        self.actionHistory.append(self.position)        
        self.capitalHistory.append(self.capital)
        self.virtualCapitalHistory.append(self.capital + sum(self.virtual))
        
        if(self.isOneToOne):
            pass
        else:
            action = map(self.discretize, self.suggester.getActions(state))
            
        self.actionize(action)
        
        
        
				
    def getVirtualCapital(self):
        return self.capital + sum(self.virtual)
        
    def reset(self):
        self.suggester.reset()
        self.virtual = [0] * self.n_Stock
        self.position = [0] * self.n_Stock
        self.actionHistory = []
        self.capitalHistory = []
        self.virtualCapitalHistory = []
        self.capital = self.resetCapital