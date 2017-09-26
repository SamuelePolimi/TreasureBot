"""
This class integrates with The Rock Trading APIs v1 and provides basic market operation.
"""

import requests
import time
import hmac
import hashlib
import json


class TheRockTradingBroker:

    endPointUrl = 'https://api.therocktrading.com/v1/'

    def __init__(self, api_key, secret):
        """
        Initialize the broker.

        Args:
            api_key (str): The api key to access the platform.
            secret (str): Needed to sign an authenticated request.
        """
        self.api_key = api_key
        self.secret = secret

    # Account API

    def balance_per_currency(self, currency):
        """
        Get the balance in a specific currency.

        Args:
            currency (str): The currency ('EUR', 'USD', BTC', 'LTC', 'XRP', ...)
        Returns:
            json: The json representation of the balance in the specified currency.
        """
        url = self.endPointUrl + 'balances/'+currency.upper()
        return requests.get(url, headers=self.get_headers(url, auth_required=True), timeout=10).json()

    def balances(self):
        """
        Get a list of all the balances in any currency.

        Returns:
            json: The json representation of the balance in any currency.
        """
        url = self.endPointUrl + 'balances'
        return requests.get(url, headers=self.get_headers(url, auth_required=True), timeout=10).json()

    # Market API

    def fund(self, market_id):
        """
        Get single market data.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
        Returns:
            json: The json representation of the market specified.
        """
        url = self.endPointUrl + 'funds/'+market_id.upper()
        return requests.get(url, headers=self.get_headers(url, auth_required=False), timeout=10).json()

    def funds(self):
        """
        Get all markets at once.

        Returns:
            json: The json representation of all the markets.
        """
        url = self.endPointUrl + 'funds'
        return requests.get(url, headers=self.get_headers(url, auth_required=False), timeout=10).json()

    def orderbook(self, market_id):
        """
        Get the entire set of bids and asks of a particular currency pair market.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
        Returns:
            json: The json representation of the orderbook related to the specified market.
        """
        url = self.endPointUrl + 'funds/'+market_id.upper()+'/orderbook'
        return requests.get(url, headers=self.get_headers(url, auth_required=False), timeout=10).json()

    def ticker(self, market_id):
        """
        Get ticker of a choosen market.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
        Returns:
            json: The json representation of the ticker related to the specified market.
        """
        url = self.endPointUrl + 'funds/'+market_id.upper()+'/ticker'
        return requests.get(url, headers=self.get_headers(url, auth_required=False), timeout=10).json()

    def tickers(self):
        """
        Get all tickers at once.

        Returns:
            json: The json representation of all the tickers.
        """
        url = self.endPointUrl + 'funds/tickers'
        return requests.get(url, headers=self.get_headers(url, auth_required=False), timeout=10).json()

    def trades(self, market_id):
        """
        Get last 200 trades for a choosen market.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
        Returns:
            json: The json representation of the trades related to the specified market.
        """
        url = self.endPointUrl + 'funds/'+market_id.upper()+'/trades?per_page=200'
        return requests.get(url, headers=self.get_headers(url, auth_required=False), timeout=10).json()

    # Market API

    def cancel_all_open_orders(self, market_id):
        """
        Remove all active orders from the specified market.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
        Returns:
            json: The platform's server response (200 response if ok).
        """
        url = self.endPointUrl + 'funds/'+market_id.upper()+'/orders/remove_all'
        return requests.delete(url, headers=self.get_headers(url, auth_required=True), timeout=10)

    def cancel_order(self, market_id, order_id):
        """
        Remove all active orders from the specified market.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
            order_id (str): The order ID
        Returns:
            json: The json representation of the removed order.
        """
        url = self.endPointUrl + 'funds/'+market_id.upper()+'/orders/'+order_id
        return requests.delete(url, headers=self.get_headers(url, auth_required=True), timeout=10).json()

    def orders(self, market_id, side=None, status=None):
        """
        List your orders for the specified market.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
            side (:obj:'str', optional): Filter orders by side. Accepted values are: 'buy', 'sell'.
            status (:obj:'str', optional): Filter orders by status. Accepted values are: 'active', 'conditional', 'executed'.
        Returns:
            json: Array of active and conditional personal orders.
        """
        url = self.endPointUrl + 'funds/'+market_id.upper()+'/orders'

        if side and not status:
            url += '?side='+side
        if not side and status:
            url += '?status='+status
        if side and status:
            url += '?side='+side+'&status='+status

        return requests.get(url, headers=self.get_headers(url, auth_required=True), timeout=10).json()

    def place_order(self, market_id, side, amount, price):
        """
        Place an order on the specified market, at specified conditions.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
            side (str): 'buy', 'sell', 'close_long', 'close_short' order.
            amount (str): The amount you want to buy or sell.
            price (str): The price of your order to be filled.
        Returns:
            json: On success returns the json representation of the order.
        """
        url = self.endPointUrl + 'funds/'+market_id.upper()+'/orders'

        post_data = {
            'fund_id': market_id.upper(),
            'side': side,
            'amount': amount,
            'price': price
        }

        return requests.post(url, data=json.dumps(post_data), headers=self.get_headers(url, auth_required=True), timeout=10).json()

    def show_order(self, market_id, order_id):
        """
        List your orders for the specified market.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
            order_id (str): The order ID
        Returns:
            json: The json representation of the removed order.
        """
        url = self.endPointUrl + 'funds/'+market_id.upper()+'/orders/'+order_id
        return requests.get(url, headers=self.get_headers(url, auth_required=True), timeout=10).json()

    def user_trades(self, market_id):
        """
        Get first 200 user's trades.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
        Returns:
            json: The json representation of the removed order.
        """
        url = self.endPointUrl + 'funds/'+market_id.upper()+'/trades?per_page=200'
        return requests.get(url, headers=self.get_headers(url, auth_required=True), timeout=10).json()

    # Helper functions

    def get_headers(self, url, auth_required=False):
        headers = {'content-type': 'application/json'}
        if auth_required:
            nonce = str(int(time.time() * 1000000))
            signature = hmac.new(self.secret, msg=(nonce+url), digestmod=hashlib.sha512).hexdigest()
            headers.update({'X-TRT-KEY': self.api_key, 'X-TRT-SIGN': signature, 'X-TRT-NONCE': nonce})

        return headers


import pprint
brokerTest = TheRockTradingBroker('4e7acaa8856b38818f80b19282ee8e32dc9a0580', 'c71e3cccac1bcf81fb091baef14e86e06d779667')
pprint.pprint(brokerTest.funds())
