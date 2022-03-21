import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

fname = input('Enter scan data file name: ')
if not fname:
    exit(0)

df = pd.read_csv('%s'%fname, index_col=0)

devlist = df.name.unique()
dflist=[]

for dev in devlist:
    dfdev= df[df.name==dev][['stamp','data']]
    dfdev['stamp']= pd.to_datetime(dfdev['stamp'])
    dfdev['TS']=dfdev['stamp']
    dfdev.rename(columns={'data':dev, 'TS':'%s Timestamp'%dev},inplace=True)
    dfdev.set_index('stamp').reset_index(drop=False, inplace=True)
    dflist.append(dfdev)

ddf = reduce(lambda  left,right: pd.merge_asof(left,right,on=['stamp'],direction='nearest',tolerance=pd.Timedelta('10ms')), dflist)
ddf.drop(columns=['stamp'], inplace=True)
print( ddf.head(10) )

ddf.to_csv('%s.csv'%(fname.strip('_raw.csv')),index_label='idx')
#ddf.dropna(inplace=True)
#plt.plot(ddf['L:V5QSET'][1:],ddf['L:D7LMSM'][1:])
#plt.show()
