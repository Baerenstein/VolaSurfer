Volatility Engine
===============

The VolatilityEngine class is responsible for managing volatility surface calculations and analysis for option markets.

.. automodule:: core.VolatilityEngine
   :members:
   :undoc-members:
   :show-inheritance:

Core Components
-------------

Data Structures
~~~~~~~~~~~~~

.. autoclass:: core.VolatilityEngine.VolPoints
   :members:
   :undoc-members:
   :special-members: __init__, __str__

Main Engine
~~~~~~~~~~

.. autoclass:: core.VolatilityEngine.VolatilityEngine
   :members:
   :undoc-members:
   :special-members: __init__

Surface Calculations
~~~~~~~~~~~~~~~~~

.. automethod:: core.VolatilityEngine.VolatilityEngine.get_volatility_surface
.. automethod:: core.VolatilityEngine.VolatilityEngine.get_skews
.. automethod:: core.VolatilityEngine.VolatilityEngine._get_term_structure

Analytics
~~~~~~~~

.. automethod:: core.VolatilityEngine.VolatilityEngine.get_implied_volatility_index
.. automethod:: core.VolatilityEngine.VolatilityEngine.get_surface_metrics

Infrastructure
==============

Core Functions
------------

.. automodule:: core.volatility
   :members:
   :undoc-members:

Helper Functions
--------------

.. automodule:: core.volatility.helpers
   :members: