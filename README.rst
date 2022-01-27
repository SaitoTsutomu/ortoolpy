`ortoolpy` is a package for Operations Research.
It is user's responsibility for the use of `ortoolpy`.

::

   from ortoolpy import knapsack
   size = [21, 11, 15, 9, 34, 25, 41, 52]
   weight = [22, 12, 16, 10, 35, 26, 42, 53]
   capacity = 100
   knapsack(size, weight, capacity)

Show Table
----------

::

   import ortoolpy.optimization
   %typical_optimization

Requirements
------------
* Python 3, pandas, pulp, more-itertools

Features
--------
* This is a sample. So it may not be efficient.
* `ortools_vrp` using Google OR-Tools ( https://developers.google.com/optimization/ ).

Setup
-----
::

   $ pip install ortoolpy

History
-------
0.0.1 (2015-6-26)
~~~~~~~~~~~~~~~~~~
* first release
