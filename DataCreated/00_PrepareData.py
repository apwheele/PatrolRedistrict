'''
This preps the data for the problem
See all the steps in the original 
notebook
'''

import dbfread
import pandas
import os
import math
import csv
import pickle
from sklearn import linear_model

#Changing the directory
MyLoc = r'D:\Dropbox\Dropbox\PublicCode_Git\PatrolRedistrict\PatrolRedistrict\DataCreated'
os.chdir(MyLoc)
print(os.getcwd())

#Get the Reporting Areas table, include call weights and XY coordinates
ra_data = []
vars = ['PDGrid','NumberId','XFeet','YFeet','Cnt_ra','Sum_Minute','Sum_Minu_1']

for record in dbfread.DBF('RA_Centroids.dbf'):
    temp = [record[i] for i in vars]
    ra_data.append(temp)

#I need to make a dictionary from this that maps the PDGrid to the NumberId value
nid_map = {}
nid_invmap = {}
cont_dict = {} #contiguity dictionary
call_dict = {} #call dictionary
dist_dict = {} #distance dictionary
areas = []
pair_data = []
for i in ra_data:
    areas.append(i[0])
    nid_map[i[1]] = i[0]
    nid_invmap[i[0]] = i[1]
    cont_dict[i[0]] = []
    call_dict[i[0]] = i[4] #could also use weights, ra_data[-1]
    dist_dict[i[0]] = {}
    for j in ra_data:
        dist_dict[i[0]][j[0]] = 9999
        pair_data.append((i+j))

#Get the contiguity matrix, add in the missing links
for record in dbfread.DBF('RA_ContMat.dbf'):
    FromRA = nid_map[record['NUMBERID']]
    ToRA = nid_map[record['NID']]
    cont_dict[FromRA].append(ToRA)

#Now I have a two more missing links to append
#1326 <-> 1333
E1326 = nid_map[1326]
E1333 = nid_map[1333]
cont_dict[E1326].append(E1333)
cont_dict[E1333].append(E1326)

#Get the distance matrix
dist_data = []
for record in dbfread.DBF('RA_DistMat.dbf'):
    f,t = record['Name'].split(' - ')
    min = record['Total_Minu']
    dist_data.append((f,t,min))

#Now creating a pandas dataframe of the pairs data
FullPairs = pandas.DataFrame(pair_data, columns=['FromRA','FromNID','FX','FY','CF','CWF','CWF2',
                                                 'ToRA',  'ToNID','TX','TY','CT','CWT','CWT2'])

#Eliminate Columns I dont need
FullPairs = FullPairs[['FromRA','FromNID','FX','FY','ToRA','ToNID','TX','TY','CT','CWT2']]
    
#Adding in column for Euclidean distance
FullPairs['EuDist'] = ( (FullPairs.FX - FullPairs.TX)**2 + (FullPairs.FY - FullPairs.TY)**2 )**0.5
    
#Now creating a pandas dataframe of the dist data, estimate linear regression
DistPairs = pandas.DataFrame(dist_data, columns = ['FromRA','ToRA','Min'])
DistPairs = DistPairs.merge(FullPairs[['FromRA','ToRA','EuDist']], on=['FromRA','ToRA'], how='left')
reg = linear_model.LinearRegression()
reg.fit(DistPairs[['EuDist']],DistPairs[['Min']])
reg.coef_, reg.intercept_
#Can see that they are highly correlated
DistPairs.corr()
#DistPairs.plot.scatter('EuDist','Min')

#Now merging the two together
MergeData = FullPairs.merge(DistPairs[['FromRA','ToRA','Min']], on=['FromRA','ToRA'], how='outer')
#You can see some missing data in the last column
MergeData['PredMin'] = reg.predict(MergeData[['EuDist']])


def Mis(row):
    if math.isnan(row['Min']):
        val = row['PredMin']
    else:
        val = row['Min']
    return val

MergeData['FinMin'] = MergeData.apply(Mis, axis=1)

#Now I can create the distance matrix 
#SubData = MergeData[:10]
for index, row in MergeData.iterrows():
    dist_dict[row['FromRA']][row['ToRA']] = row['FinMin']

# Saving the objects I need

#areas
#dist_dict
#cont_dict
#call_dict
#nid_invmap
#MergeData

fin_object = [areas, dist_dict, cont_dict, call_dict, nid_invmap, MergeData]
pickle.dump(fin_object, open("fin_obj.pkl", "wb" ))
