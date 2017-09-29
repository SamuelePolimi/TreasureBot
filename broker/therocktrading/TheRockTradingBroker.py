"""
This class integrates with The Rock Trading APIs v1 and provides basic market operation.
"""

import requests
import time
import hmac
import hashlib
import json
import math
import numpy as np
from datetime import datetime

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

    def last_trade(self, market_id):
        """
        Get last trade for a choosen market.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
        Returns:
            json: The json representation of the trades related to the specified market.
        """
        url = self.endPointUrl + 'funds/'+market_id.upper()+'/trades?per_page=1'
        return requests.get(url, headers=self.get_headers(url, auth_required=False), timeout=10).json()

    def trades(self, market_id, before, after, file_name):
        """
        Get last trades for a choosen market for a specific interval.

        Args:
            market_id (str): The market ID ('BTCEUR', 'BTCUSD', PPCBTC', 'LTCEUR', ...)
            before (str): Get only trades executed after a certain timestamp ( format %Y-%m-%dT%H:%M:%S%Z ex. 2015-02-06T08:47:26Z )
            after (str): Get only trades executed after a certain timestamp ( format %Y-%m-%dT%H:%M:%S%Z ex. 2015-02-06T08:47:26Z )
        Returns:
            json: The json representation of the trades related to the specified market.
        """
        results_per_page = 200
        page_index = 1
        last_page = 1

        np.save(file_name,{'date':np.zeros([0]),'amount':np.zeros([0]),'price':np.zeros([0])})
        while page_index <= last_page:
            url = self.endPointUrl + 'funds/' + market_id.upper() + '/trades?per_page=' + str(
                results_per_page) + '&before=' + before + '&after=' + after + '&page=' + str(page_index)
            request = requests.get(url, headers=self.get_headers(url, auth_required=False), timeout=10).json()

            # fire the first request and get metadata and first page of results
            if page_index == 1:
                last_page = int(math.ceil(float(request['meta']['total_count']) / results_per_page))
                print "Number of pages to download:", last_page
                print "Progress 0/" + str(last_page)+ ": .",
            if page_index%10==0:
                print "\nProgress " + str(page_index) + "/" + str(last_page)+ ":",
            print ".",
            #Get the list of prices
            trade_list = request['trades']
            numpy_file = np.load(file_name)
            prew_price = numpy_file.item().get('price')
            prew_date = numpy_file.item().get('date')
            prew_amount = numpy_file.item().get('amount')

            price_list = map(lambda x: x['price'], trade_list)
            price_list.reverse()

            #Get the dates
            date_list = map(lambda d: datetime.strptime(d['date'],"%Y-%m-%dT%H:%M:%S.%fZ"), trade_list)
            date_list.reverse()

            #Get the volume
            amount_list  = map(lambda a: a['amount'], trade_list)
            amount_list.reverse()

            np.save(file_name,{'price':np.array(price_list+prew_price.tolist()),
                    'amount':np.array(amount_list+prew_amount.tolist()),
                    'date':np.array(date_list+prew_date.tolist())})

            page_index += 1
        content = np.load(file_name)
        famount = content.item().get('amount')
        fdate = content.item().get('date')
        fprice = content.item().get('price')
        fdt = map(lambda x: (x[1]-x[0]).total_seconds(), zip(fdate[:-1],fdate[1:]))
        np.save(file_name,{'amount':famount,'price':fprice,'time':np.array(fdt)})
        print ". End."


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
            signature = hmac.new(self.secret.encode(), msg=(nonce+url).encode(), digestmod=hashlib.sha512).hexdigest()
            headers.update({'X-TRT-KEY': self.api_key, 'X-TRT-SIGN': signature, 'X-TRT-NONCE': nonce})

        return headers
