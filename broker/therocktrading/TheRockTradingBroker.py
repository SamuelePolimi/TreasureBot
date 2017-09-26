"""
This class integrates with The Rock Trading APIs v1 and provides basic market operation.
"""

class TheRockTradingBroker:

    def __init__(self, apiKey, secret):
        """
        Initialize the broker
        :param apiKey: The api key to access the platform
        :type secret: needed to sign an authenticated request
        """
