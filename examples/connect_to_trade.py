"""
In this example we connect to our broker and we perform some request.
It is supposed to have a credentials.json file in this folder.
credentials.json has the following structure
{"KEY":"mykey",
"SECRET":"mysecret"
}
"""
import json
import pprint
from broker.therocktrading.TheRockTradingBroker import TheRockTradingBroker

with open('credentials.json') as json_data:
    credential = json.load(json_data)

therock = TheRockTradingBroker(credential['KEY'],credential['SECRET'])


pprint.pprint(therock.funds())
pprint.pprint(therock.tickers())
pprint.pprint(therock.orderbook("BTCEUR"))
pprint.pprint(therock.user_trades("BTCEUR"))
pprint.pprint(therock.orders("BTCEUR"))