import numpy as np
from scipy.signal import find_peaks
import pylab as plt
from matplotlib.pyplot import figure

def peak_finder(x,t,height,prominence,distance):
    peaksx, _ = find_peaks(x, height=height,prominence=prominence,distance=distance)
    pxx = []
    pxt = []
    for xx in peaksx:
        pxt.append(t[int(xx)])
        pxx.append(x[int(xx)])

    pxx = np.asarray(pxx)
    pxt = np.asarray(pxt)

    dtx =[]
    for i in range (1,len(pxx)):
        dtx.append(pxt[i] - pxt[i-1])
    dtx = np.asarray(dtx)

    return (pxt,pxx,dtx)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    #if array[idx]<value and idx<array.size-1:
    #    idx=idx+1
    return (idx,array[idx])

def align_peaks_dt(tx,xx,ty,yy,dt):
    tyloc=ty-dt
    dt=0
    n=0
    for i,txi in enumerate(tx):
        (idx,val)=find_nearest(tyloc,txi)
        if abs(txi-val)<4.5e-9:
            dt+=txi-val
            n+=1

    return dt/n

def compare_peaks(tx,xx,ty,yy,dt):
    tyloc=ty-dt
    sum=0
    n=0
    for i,txi in enumerate(tx):
        (idx,val)=find_nearest(tyloc,txi)
        if abs(txi-val)<1e-9:
            sum+=(xx[i]-yy[idx])*(xx[i]-yy[idx])
            n+=1
    if n==0:
        n=1
    return (n,sum/n)


init_heightx=(0.3,1)
init_heighty=(0.3,1)
init_dist=60
init_prom=(0.25,1)

fill=None
t = []
x = [] 
y = [] 
z = []

path = r'/Users/rshara01-local/Desktop/Griffin/Study 12302021/2study12302021/'
#path = r'/Users/rshara01-local/Desktop/Winter 2022/Study 12302021/'
suffix = '_ALL.csv'
filename = path + r'3back3' + suffix

file = open('%s'%filename,encoding="utf-8")

lines = file.readlines()[12:]

for line in lines:
    cols = [j for j in line.split(',')]
    t.append(float(cols[0]))
    x.append(float(cols[1]))
    y.append(float(cols[2]))
    #z.append(float(cols[3]))
    
t = np.asarray(t)
x = np.asarray(x)
y = np.asarray(y)
x[x<0]=0
y[y<0]=0
#x=-x
#y=-y

#for 3back and above: swap x and y
y_norm=(x-x.min(0))/x.ptp(0)
x_norm=(y-y.min(0))/y.ptp(0)

(pxt,pxx,dtx)=peak_finder(x_norm,t,init_heightx,init_prom,init_dist)
(pyt,pyy,dty)=peak_finder(y_norm,t,init_heighty,init_prom,init_dist)

figure(figsize=(10,6))
plt.plot(t,x_norm)
plt.plot(pxt,pxx,marker='o',linewidth=0)
plt.plot(t,y_norm)
plt.plot(pyt,pyy,marker='o',linewidth=0)
plt.xlim([1.96e-6,2.22e-6])
plt.show()

bestdt=0
nbest=0
mindsig=1
for irough in range(100):
    dt_rough=irough*5e-9
    dt=align_peaks_dt(pxt,pxx,pyt,pyy,dt_rough)
    (n,dsig)=compare_peaks(pxt,pxx,pyt,pyy,(dt_rough-dt))
    print("shifting peaks by",dt_rough-dt,n,dsig)
    if n>100 and dsig<mindsig:
        mindsig=dsig
        bestdt=dt_rough-dt
        nbest=n

    if (irough<3):
        figure(figsize=(10,6))
        plt.plot(t,x_norm)
        plt.plot(pxt,pxx,marker='o',linewidth=0)
        plt.plot(t-(dt_rough-dt),y_norm)
        plt.plot(pyt-(dt_rough-dt),pyy,marker='o',linewidth=0)
        plt.xlim([1.96e-6,2.22e-6])
        plt.show()

figure(figsize=(10,6))
print ("Best shift",bestdt,mindsig,nbest)
plt.plot(t,x_norm)
plt.plot(pxt,pxx,marker='o',linewidth=0)
plt.plot(t-bestdt,y_norm)
plt.plot(pyt-bestdt,pyy,marker='o',linewidth=0)
plt.show()
