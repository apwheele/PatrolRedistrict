# Patrol Redistricting in Python
 
This is my work on using a p-median model to redistrict patrol areas in Carrollton, TX. In particular, this incorporates workload inequality constraints into the p-median model.

The final published paper is:

> Wheeler, Andrew P. (2019) Creating optimal patrol areas using the P-median model. [*Policing: An International Journal* 42(3): 318-333](https://doi.org/10.1108/PIJPSM-02-2018-0027).

You can also access a pre-print version here or on [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3109791).

The Jupyter notebook shows how to use the python `pulp` library to set up the linear programming problem. And ArcGIS and the network extension to get the travel time distances and weights needed to optimize areas.