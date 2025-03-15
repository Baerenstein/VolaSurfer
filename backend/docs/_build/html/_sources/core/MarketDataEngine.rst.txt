Market Data Engine
==================

The MarketDataEngine class is responsible for collecting, processing, and storing market data for cryptocurrency options.

.. automodule:: core.MarketDataEngine
   :members:
   :undoc-members:
   :show-inheritance:

Main Components
--------------

Initialization
~~~~~~~~~~~~~~
.. automethod:: core.MarketDataEngine.MarketDataEngine.__init__

Market Data Processing
~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: core.MarketDataEngine.MarketDataEngine.process_market_updates
.. automethod:: core.MarketDataEngine.MarketDataEngine._process_currency_updates

Instrument Management
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: core.MarketDataEngine.MarketDataEngine.initialize_instruments
.. automethod:: core.MarketDataEngine.MarketDataEngine._get_options_chain

Market State
~~~~~~~~~~~~
.. automethod:: core.MarketDataEngine.MarketDataEngine.get_last_price
.. automethod:: core.MarketDataEngine.MarketDataEngine._check_market_state

Volatility Surface
~~~~~~~~~~~~~~~~~~
.. automethod:: core.MarketDataEngine.MarketDataEngine._get_vol_surface

Data Storage
~~~~~~~~~~~~
.. automethod:: core.MarketDataEngine.MarketDataEngine._store_data

Error Handling
~~~~~~~~~~~~~~
.. automethod:: core.MarketDataEngine.MarketDataEngine._handle_error

Main Loop
~~~~~~~~~
.. automethod:: core.MarketDataEngine.MarketDataEngine.run
