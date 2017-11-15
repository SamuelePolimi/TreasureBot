"""
In this example we connect to our broker and we perform some request.
It is supposed to have a credentials.json file in this folder.
credentials.json has the following structure
{"KEY":"mykey",
"SECRET":"mysecret"
}
"""
from datetime import datetime
import json
import numpy as np
import pprint
from broker.therocktrading.TheRockTradingBroker import TheRockTradingBroker

with open('credentials.json') as json_data:
    credential = json.load(json_data)

therock = TheRockTradingBroker(credential['KEY'],credential['SECRET'])

therock.trades('BTCEUR', '2017-09-28T20:21:59.000Z', '2017-09-26T12:00:00.000Z',"puppy.npy")
print np.load("puppy.npy")
