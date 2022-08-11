import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = 'cm'

t = [None]*4
x = [None]*4 
xplat = [None]*4

med = [None]*4
dt  = [219.782, 220.010, 220.243, 220.487] # external input: from finddt.py

path = r'/Users/rshara01-local/Desktop/Winter 2022/Study 12302021/2study12302021/'
suffix = '_ALL.csv'


for i in range(4):    
    filename = path + r'%dfront3'%i + suffix
    file = io.open('%s'%filename,encoding="utf-8")
    lines = file.readlines()[12:]

    t[i] = []
    x[i] = []

    for line in lines:
        cols = [j for j in line.split(',')]
        t[i].append(float(cols[0]))
        x[i].append(float(cols[1]))

    
    t[i] = np.asarray(t[i])
    x[i] = np.asarray(x[i])


t[0] = t[0]+1.7e-6
for i in range(4):
    xplat[i] = [x[i][j] for j,t in enumerate(t[i]) if t >0 and t<28e-6]
    med[i] = np.median(xplat[i])

    print(med[i])

nmz = ['8nsboth2','1back3','2back1','3back1'] #1back1 1back2
labels = ['No delay','2.125\"','4.25\"','6.375\"']
filenm = ['./files/%s.npz'%nmz[i] for i in range(len(nmz))]
data = [np.load(f) for f in filenm]

dT =[None]*len(nmz)

for i,d in enumerate(data):
    dT[i] = []
    for j in range (min(len(d['pxx']), len(d['pyy']))):
        dT[i].append(d['pyt'][j] - d['pxt'][j])
    dT[i] = np.asarray(dT[i])    
    dT[i] = np.absolute(dT[i])


plt.xlabel('# Trace peaks',fontsize='xx-large')
plt.ylabel('$\Delta$T (ns)',fontsize='xx-large')
plt.xticks(fontsize='x-large')
plt.yticks(fontsize='x-large')
plt.grid(color='k', linestyle='-', linewidth=2)
[ plt.plot(dT[i]/1.e-9, label='%s'%labels[i]) for i in range(len(dT))]
plt.legend(fontsize='x-large')
plt.show()

plt.title("Phase det. signal",fontsize='xx-large')
plt.xlabel('Time ($\mu$)s',fontsize='xx-large')
plt.ylabel('Signal (V)',fontsize='xx-large')
plt.xticks(fontsize='x-large')
plt.yticks(fontsize='x-large')
plt.grid(color='k', linestyle='-', linewidth=2)
[plt.plot(t[i]*1e6, x[i], marker = '',label='%s'%labels[i]) for i in range(4)]
plt.legend(fontsize='x-large')
plt.show()

plt.grid(color='k', linestyle='-', linewidth=2)
[plt.plot(xplat[i], marker = '',label='') for i in range(4)]
plt.show()

terr = [0.08]*4
coef = np.polyfit(dt,med,1,w=1/terr)
poly1d_fn = np.poly1d(coef) 

plt.title("Griffin calibration")
plt.xlabel('dt G1-G2 (ns)')
plt.ylabel('phase det. signal (V)')
plt.grid(color='k', linestyle='-', linewidth=2)
plt.plot(dt,med, 'bo', dt, poly1d_fn(dt), '--b')
plt.errorbar(dt,med, xerr=terr, linestyle='', c='b')
#plt.plot(dt,med)
plt.show()
print(coef)
