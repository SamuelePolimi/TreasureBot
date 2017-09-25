import time

"""
This class represents an order.
"""

class Order:

    def __init__(self, sell=False, quantity=0, amount=0., expiration=None):
        """
        Initialize a Order
        :param sell: True if sell, buy otherwise
        :type sell: bool
        :param quantity: the quantity of currency to trade
        :type quantity: float
        :param amount: the price of selling or buying
        :type amount: float
        :param expiration: the expiration date. If None, there is no expiration.
        :type expiration: time.strcut_time
        """
        self.sell = sell
        self.quantity = quantity
        self.amount = amount
        self.expiration = expiration

    def open(self):
        raise Exception("Not implemented yet")

    def close(self):
        raise Exception("Not implemented yet")