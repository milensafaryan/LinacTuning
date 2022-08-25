import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.ticker as mt
from functools import reduce

plt.rcParams["axes.titlelocation"] = 'right'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["savefig.format"] = 'pdf'


file = './Linac_current_5years.xls'

#df = pd.read_csv(file,sep='\t', nrows=20000)
df = pd.read_csv(file,sep='\t',low_memory=False)

dflist = []

for i in range(int(len(df.keys())/2)):
    subdf=df.iloc[:,2*i:2*i+2].replace(r'^\s*$', np.nan, regex=True)
    subdf.dropna(inplace=True)
    subdf[subdf.columns[0]]=subdf[subdf.columns[0]].str.replace('_',' ')
    subdf[subdf.columns[0]]=subdf[subdf.columns[0]].str.replace(';',':')
    subdf[subdf.columns[0]]=pd.to_datetime(subdf[subdf.columns[0]])
    subdf[subdf.columns[1]]=pd.to_numeric(subdf[subdf.columns[1]])
    subdf['TS %s'%subdf.columns[1]]= subdf[subdf.columns[0]]
    subdf.rename(columns={subdf.columns[0]:'Time'},inplace=True)
    
    dflist.append(subdf)


ddf = reduce(lambda  left,right: pd.merge_asof(left.sort_values('Time'),right.sort_values('Time'),on=['Time'],direction='nearest',tolerance=pd.Timedelta(minutes=6)), dflist)
ddf.drop(columns=['Time'], inplace=True)

ddf.loc[ddf['L|SOURCE(B7)']>0,'SourceA']=ddf['L:ATOR']
ddf.loc[ddf['L|SOURCE(B7)']>0,'TS SourceA']=ddf['TS L:ATOR']
ddf.loc[ddf['L|SOURCE(B6)']>0,'SourceB']=ddf['L:BTOR']
ddf.loc[ddf['L|SOURCE(B6)']>0,'TS SourceB']=ddf['TS L:BTOR']

#ddf.loc[:,:] = ddf.loc[(ddf['SourceAB']>0) & ((ddf['L:ATOR']>0) | (ddf['L:BTOR']>0)) & (ddf['L:D7TOR']>0)]

ylist=['SourceA','SourceB','L:TO1IN','L:D7TOR']
lbl = ['Source A','Source B','Linac input', 'Linac output']
colors = ['r','b','c','g']
lims = [[0,80],[0,80],[0,80],[0,80]]

fig = plt.figure(figsize=(12,7))
p = [None]*len(ylist)
ax = [None]*len(ylist)
ax[0] = fig.add_subplot(111)
ax[0].xaxis.grid(True, which='major')
ax[0].yaxis.grid(True, which='major')
#ax[0].set_xlabel('Date', fontsize='xx-large')
ax[0].set_ylabel('Beam current (mA)', fontsize='xx-large')
ax[0].tick_params(axis='x',rotation=45)
date_form = DateFormatter("%Y-%m-%d")
ax[0].xaxis.set_major_formatter(date_form)
ax[0].xaxis.set_major_locator(MonthLocator(interval=6))

for i in range(1,len(ylist)):
    ax[i] = ax[0].twinx()

for i,yd in enumerate(ylist):
    ax[i].tick_params(axis='y', colors=colors[i],rotation=0)
    ax[i].tick_params(axis='both',labelsize='xx-large')
    ax[i].yaxis.set_major_locator(mt.LinearLocator(5))
    p[i] = ax[i].scatter(ddf['TS %s'%yd],ddf[yd],marker='.',s=1,linestyle='None',c=colors[i],label=lbl[i])
    ax[i].set_ylim(lims[i])
    if i in [0,1]:
        ax[i].yaxis.tick_left()
    else:
        ax[i].yaxis.tick_right()
    for yl in ax[i].get_yticklabels():
        if i%2==0:
            yl.set(verticalalignment='bottom')
        else:
            yl.set(verticalalignment='top')
        

leg = fig.legend(loc='upper right',bbox_transform=ax[0].transAxes, bbox_to_anchor=[1.009, 1.018], fontsize='x-large',fancybox=True, framealpha=0.9, handlelength=0, handletextpad=0)

for h, t in zip(leg.legendHandles, leg.get_texts()):
    h.set_visible(False)
    t.set_color(h.get_facecolor()[0])
    
plt.subplots_adjust(wspace=0, hspace=0)
plt.subplots_adjust(bottom=0.2)
plt.subplots_adjust(top=0.9)
plt.subplots_adjust(left=0.1)
plt.subplots_adjust(right=0.90)

plt.show()

