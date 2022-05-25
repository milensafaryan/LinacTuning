import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, find_peaks_cwt, peak_widths

t = [None]*4
x = [None]*4 
xplat = [None]*4

med = [None]*4
dt  = [219.782, 220.010, 220.243, 220.487] # external input: from fit

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

'''
x_normed = (x - x.min(0)) / x.ptp(0)
peaksx, _ = find_peaks(x_normed, height=0.717, distance=31 ,prominence=0.27)
pxx = []
pxt = []
for xx in peaksx:
    pxt.append(t[int(xx)])
    pxx.append(x_normed[int(xx)])

pxt = np.asarray(pxt)
pxx = np.asarray(pxx)
'''

nmz = ['1back1','1back2','1back3','2back1','3back1','8nsboth2']
filenm = ['./files/%s.npz'%nmz[i] for i in range(len(nmz))]
data = [np.load(f) for f in filenm]

print(data[0]['pxx'][0])

dT =[None]*len(nmz)

for i,d in enumerate(data):
    dT[i] = []
    for j in range (min(len(d['pxx']), len(d['pyy']))):
        dT[i].append(d['pyt'][j] - d['pxt'][j])
    dT[i] = np.asarray(dT[i])    
    dT[i] = np.absolute(dT[i])


plt.xlabel('peaks')
plt.ylabel('time diff (8e-11)')
plt.grid(color='k', linestyle='-', linewidth=2)
[ plt.plot(dT[i]/0.8e-10, label='%s'%nmz[i]) for i in range(len(dT))]
plt.legend()
plt.show()

plt.title("Phase det. signal")
plt.xlabel('T')
plt.ylabel('V')
plt.grid(color='k', linestyle='-', linewidth=2)
[plt.plot(t[i], x[i], marker = '',label='') for i in range(4)]
plt.show()

plt.grid(color='k', linestyle='-', linewidth=2)
[plt.plot(xplat[i], marker = '',label='') for i in range(4)]
plt.show()

terr = [0.08]*4
coef = np.polyfit(dt,med,1,w=terr)
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

'''
plt.grid(color='k', linestyle='-', linewidth=2)
delta = []
for i in range (peaksx[0],peaksx[0]+1000):
    delta.append( -x[i] + y[i-shift])

plt.plot(t[peaksx[0]:peaksx[0]+1000],y_normed[peaksx[0]-172:peaksx[0]+1000-172], c='b')
plt.plot(t[peaksx[0]:peaksx[0]+1000],x_normed[peaksx[0]:peaksx[0]+1000], c='r')
plt.plot(t[peaksx[0]:peaksx[0]+1000], delta[:], c='g')
plt.plot(pyt[:], pyy[:], marker='x', linewidth=0, c='r', label='')
plt.plot(pxt[:], pxx[:], marker= 'x', linewidth=0, c = 'b',label='')
plt.plot(pxt[:100],pxx[:100], marker='x',linewidth=0,  c='r')
plt.plot(pxt[:100],pyy[:100], marker='x',linewidth=0,  c='b')
plt.show()

plt.scatter(delta[:],x[peaksx[0]:peaksx[0]+1000], c='m')
plt.scatter(delta[:],y[peaksx[0]-172:peaksx[0]+1000-172], c='c')
plt.scatter(pxx,pyy, c='y')
plt.ylim(-0.1,0.4)
plt.show()
'''
