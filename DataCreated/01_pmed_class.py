'''
# Here is an example creating an environment with all the stuff installed
# To run this code
conda create -n linprog
conda activate linprog
conda install -c conda-forge python=3 pip pandas numpy networkx scikit-learn dbfread geopandas glpk pyscipopt pulp
'''

import pickle
import pulp
import networkx
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd

class pmed():
    """
    Ar - list of areas in model
    Di - distance dictionary organized Di[a1][a2] gives the distance between areas a1 and a2
    Co - contiguity dictionary organized Co[a1] gives all the contiguous neighbors of a1 in a list
    Ca - dictionary of total number of calls, Ca[a1] gives calls in area a1
    Ta - integer number of areas to create
    In - float inequality constraint
    Th - float distance threshold to make a decision variables
    """
    def __init__(self,Ar,Di,Co,Ca,Ta,In,Th):
        # Assigning initial properties of object
        self.Ar = Ar
        self.Di = Di
        self.Co = Co
        self.Ca = Ca
        self.Ta = Ta
        self.In = In
        self.Th = Th
        self.subtours = [] #empty subtours to start
        self.objective = -1 #objective values
        self.pairs = None #where to stuff the matched areas
        # Creating inequality metrics
        SumCalls = sum(Ca.values())
        MaxIneq = (SumCalls/Ta)*(1 + In)
        MinIneq = (SumCalls/Ta)*(1 - In)
        self.ineq = [MaxIneq,MinIneq]
        # Creating contiguity graph
        G = networkx.Graph()
        for i in Ar:
            for j in Co[i]:
                G.add_edge(i,j)
        self.co_graph = G
        # Creating threshold vectors for decision variables
        NearAreas = {}
        Thresh = []
        for s in Ar:
            NearAreas[s] = []
            for d in Ar:
                if Di[s][d] < Th:
                    Thresh.append((s,d))
                    NearAreas[s].append(d)
        self.NearAreas = NearAreas
        self.Thresh = Thresh
        # Setting up the pulp problem
        P = pulp.LpProblem("P-Median",pulp.LpMinimize)
        # Decision variables
        assign_areas = pulp.LpVariable.dicts("SD",
                       [(s,d) for (s,d) in Thresh], 
                       lowBound=0, upBound=1, cat=pulp.LpInteger)
        # Just setting the y_vars as the diagonal sources/destinations
        y_vars = {s:assign_areas[(s,s)] for s in Ar}
        tot_constraints = 0
        self.assign_areas = assign_areas
        self.y_vars = y_vars
        # Function to minimize
        P += pulp.lpSum(Ca[d]*Di[s][d]*assign_areas[(s,d)] for (s,d) in Thresh)
        # Constraint on max number of areas
        P += pulp.lpSum(y_vars[s] for s in Ar) == Ta
        tot_constraints += 1
        # Constraint nooffbeat if local is not assigned (1)
        # Second is contiguity constraint
        for s,d in Thresh:
            P += assign_areas[(s,d)] - y_vars[s] <= 0
            tot_constraints += 1
            if s != d:
                # Identifying locations contiguous in nearest path
                both = set(networkx.shortest_path(G,s,d)) & set(Co[d])
                # Or if nearer to the source
                nearer = [a for a in Co[d] if Di[s][a] < Di[s][d]]
                # Combining, should alwayss have at least 1
                comb = list( both | set(nearer) )
                # Contiguity constraint
                P += pulp.lpSum(assign_areas[(s,a)] for a in comb if a in NearAreas[s]) >= assign_areas[(s,d)]
                tot_constraints += 1
        # Constraint every destination covered once
        # Then Min/Max inequality constraints
        for (sl,dl) in zip(Ar,Ar):
            P += pulp.lpSum(assign_areas[(s,dl)] for s in NearAreas[dl]) == 1
            P += pulp.lpSum(assign_areas[(sl,d)]*Ca[d] for d in NearAreas[sl]) <= MaxIneq
            P += pulp.lpSum(assign_areas[(sl,d)]*Ca[d] for d in NearAreas[sl]) >= MinIneq*y_vars[sl]
            tot_constraints += 3
        self.model = P
        print(f'Total number of decision variables {len(Thresh)}')
        print(f'Total number of constraints {tot_constraints}')
        av_solv = pulp.listSolvers(onlyAvailable=True)
        print(f'Available solvers from pulp, {av_solv}')
    def write_lp(self,filename,**kwargs):
        self.model.writeLP(filname,**kwargs)
    def solve(self,solver=None):
        """
        For solver can either pass in None for default pulp, or various pulp solvers, e.g.
        solver = pulp.CPLEX()
        pulp.CPLEX_CMD(msg=True, warmStart=True)
        solver = pulp.PULP_CBC_CMD(timeLimit=1000)
        solver = pulp.GLPK_CMD()
        etc.
        run print( pulp.listSolvers(onlyAvailable=True) )
        to see available solvers on your machine
        """
        print(f'Starting to solve function at {datetime.now()}')
        if solver == None:
            self.model.solve()
        else:
            self.model.solve(solver)
        print(f'Solve finished at {datetime.now()}')
        stat = pulp.LpStatus[self.model.status]
        if stat != "Optimal":
            print(f"Status is {stat}")
            try:
                self.objective = pulp.value(self.model.objective)
                print(f'Objective value is {self.objective}, but beware not optimal')
            except:
                print('Unable to grab objective value')
        else:
            self.objective = pulp.value(self.model.objective)
            print(f"Status is optimal\ntotal weighted travel is {self.objective}")
        results = []
        try:
            for (s,d) in self.Thresh:
                if self.assign_areas[(s,d)].varValue == 1.0:
                    results.append((s,d,self.Di[s][d],self.Ca[d],self.Ca[d]*self.Di[s][d]))
            results_df = pd.DataFrame(results,columns=['Source','Dest','Dist','Calls','DWeightCalls'])
            self.pairs = results_df
            # Calculating number of unique areas as a check
            source_areas = pd.unique(results_df['Source'])
            tot_source = len(source_areas)
            if tot_source == self.Ta:
                print(f'Total source areas is {tot_source}, as you specified')
            else:
                print(f'Potential Error, total source areas is {tot_source}, specified {self.Ta} areas')
        except:
            print('Unable to append results')
    def map_plot(self,geo_map,id_str,savefile=None):
        # Merging in data into geoobject
        geo_mer = geo_map.merge(self.pairs, left_on=id_str, right_on='Dest',indicator='check_merge')
        total_merge = (geo_mer['check_merge'] == 'both').sum()
        if total_merge != geo_map.shape[0]:
            print('Check the pairs/merge, not all are merged into basemap')
            print( geo_merge['check_merge'].value_counts() )
        # making centroid object for source and dissolve object
        source_locs = geo_mer[geo_mer['Source'] == geo_mer['Dest']].copy()
        diss_areas = geo_mer.dissolve(by='Source',aggfunc='sum')
        # Now making the plot
        ax = geo_mer.plot(column='Source', cmap='Spectral', categorical=True)
        source_locs.geometry.centroid.plot(ax=ax,color='k',edgecolor='white')
        diss_areas.boundary.plot(ax=ax,facecolor=None,edgecolor='k')
        if savefile:
            plt.savefig(savefile, dpi=500, bbox_inches='tight')
        else:
            plt.show()
    def collect_subtours(self):
        subtours = [] 
        areas = pd.unique(self.pairs['Source']).tolist()
        for a in areas:
            a0 = self.pairs['Source'] == a
            a1_dest = self.pairs.loc[a0,'Dest'].tolist()
            subg = self.co_graph.subgraph(a1_dest).copy()
            # Connected components
            cc = [list(c) for c in networkx.connected_components(subg)]
            # Any component that does not the source in it is a subtour
            if len(cc) == 1:
                print(f'Source {a} has no subtour')
            else:
                print(f'Source {a} has {len(cc)-1} subtours')
                for c in cc:
                    if a in c:
                        pass
                    else:
                        subtours.append((a,c))
        if len(subtours) >= 1:
            # Stats for how many calls/crimes are in those subtours
            for i,s in enumerate(subtours):
                tot_calls = 0
                for a in s[1]:
                    tot_calls += self.Ca[a]
                print(f'{i}: Subtour {s} has total {tot_calls} calls')
            # Adding subtour contraints back into main problem
            for src,des in subtours:
                sub_check = len(des) - 1
                self.model += pulp.lpSum(pmed12.assign_areas[(src,d)] for d in des) <= sub_check
            # Adding subtours into model object
            self.subtours += subtours
            # Message to warm start
            print('When resolving model, may wish to use warmStart=True if available for solver')
            return -1
        else:
            print('No subtours found, your solution appears OK')
            return 1

# Function to collect subtours
# Add subtour constraints to problem
# Redo model with warm start

###############################################################
# Loading in the data

data_loc = r'D:\Dropbox\Dropbox\PublicCode_Git\PatrolRedistrict\PatrolRedistrict\DataCreated\fin_obj.pkl'
areas, dist_dict, cont_dict, call_dict, nid_invmap, MergeData = pickle.load(open(data_loc,"rb"))

# shapefile for reporting areas
carr_report = gpd.read_file(r'D:\Dropbox\Dropbox\PublicCode_Git\PatrolRedistrict\PatrolRedistrict\DataOrig\ReportingAreas.shp')

###############################################################

###############################################################
# Fitting the models
# 3 hour time limit, note this does not count the presolve time
tl = 3*60*60

pmed12 = pmed(Ar=areas,Di=dist_dict,Co=cont_dict,Ca=call_dict,Ta=12,In=0.1,Th=10)
pmed12.solve(solver=pulp.CPLEX(timeLimit=tl,msg=True))     # takes around 10 minutes
#pmed12.solve(solver=pulp.SCIP_CMD(timeLimit=tl,msg=True)) # takes about 5 hours to get the solution
#pmed12.solve(solver=pulp.GLPK_CMD(timeLimit=tl,msg=True)) # does not converge even after 12+ hours

# Showing a map
pmed12.map_plot(carr_report, 'PDGrid')

# Figure out subtours
stres = pmed12.collect_subtours()

# Resolving with warm start with subtour constraints
pmed12.solve(solver=pulp.CPLEX_CMD(timeLimit=tl,msg=False,warmStart=True))

# Now it is OK
stres = pmed12.collect_subtours()
pmed12.map_plot(carr_report, 'PDGrid')

########################################
## To iterate and resolve if daring
#iters = 1
#while (stres == -1) or (iters < 6):
#    print('Iteration {iters} out of 5')
#    stres = pmed12.solve(solver=pulp.CPLEX_CMD(msg=False,warmStart=True))
#    pmed12.collect_subtours()
#    iters += 1
#########################################

###############################################################

