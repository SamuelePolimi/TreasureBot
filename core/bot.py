def buySell(last_action, action, state):
	#Not sure it is right
	if last_action==action:
		return 0
	if last_action!=0:
		return - last_action * state
	return last_action * state

class Bot:

    def __init__(self, finalObj):
        self.finalObj = finalObj

    """
    n_stock is a tuple of number of stock to buy/sell for each signal
    cost is a function which takes as input a tuple and give back the cost of each value
    """
    def initialize(self, n_stock, cost):
        self.n_stock = n_stock
        self.cost = cost
        self.gain = 0

    """
    WE have to well define this class
    """
    def step(self, state):
        action = self.finalObj(state)
        gain = map(buySell, action, action, state)
        self.gain += sum(gain) + sum(self.cost(state))
				
