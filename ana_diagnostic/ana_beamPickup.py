import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, find_peaks_cwt, peak_widths

t = []
x = [] 
y = [] 
z = []
t2 = []

#path = r'/Users/rshara01-local/Desktop/Winter 2022/Study 12302021/2study12302021/'
path = r'/Users/rshara01-local/Desktop/Winter 2022/Study 12302021/'
suffix = '_ALL.csv'
name = r'8nsboth2'
filename = path + name + suffix

file = io.open('%s'%filename,encoding="utf-8")

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

x_normed = (x - x.min(0)) / x.ptp(0)
y_normed = (y - y.min(0)) / y.ptp(0)

peaksx, _ = find_peaks(x_normed, height=0.645, distance=31 ,prominence=0.270)
pxx = []
pxt = []
for xx in peaksx:
    pxt.append(t[int(xx)])
    pxx.append(x_normed[int(xx)])

peaksy, _ = find_peaks(y_normed, height=0.665, distance=31 ,prominence=0.270)
pyy = []
pyt = []
for yy in peaksy:
    pyt.append(t[int(yy)])
    pyy.append(y_normed[int(yy)])

pyy = np.asarray(pyy)
pyt = np.asarray(pyt)
pxt = np.asarray(pxt)
pxx = np.asarray(pxx)


pxx = np.delete(pxx,[0,1,2,2578,2579,2580])
pxt = np.delete(pxt,[0,1,2,2578,2579,2580])

pxx = np.insert(pxx,10,0.626)
pxt = np.insert(pxt,10,1.6932e-7)

pxx = np.delete(pxx,4723)
pxt = np.delete(pxt,4723)

pxx = np.insert(pxx,5851,0.639)
pxt = np.insert(pxt,5851,3.02283e-5)

pyy = np.delete(pyy,[0,1,2,3,4,2580])
pyt = np.delete(pyt,[0,1,2,3,4,2580])

pyy = np.insert(pyy,10,0.708)
pyt = np.insert(pyt,10,-5.050e-8)

pyy = np.insert(pyy,5846,0.7)
pyt = np.insert(pyt,5846,2.99836e-5)

'''
## clean-up 3back1
pxx = np.delete(pxx,[0,1,-1,-2])
pxt = np.delete(pxt,[0,1,-1,-2])

pxx = np.insert(pxx,0,0.663)
pxt = np.insert(pxt,0,3.2987e-7)
pxx = np.insert(pxx,1,0.643)
pxt = np.insert(pxt,1,3.3492e-7)
pxx = np.insert(pxx,2,0.642)
pxt = np.insert(pxt,2,3.3985e-7)
pxx = np.insert(pxx,1320,0.679)
pxt = np.insert(pxt,1320,7.12767e-6)
pxx = np.insert(pxx,5363,0.685)
pxt = np.insert(pxt,5363,2.79238e-5)

pyy = np.delete(pyy,[0,1,2,3,-1])
pyt = np.delete(pyt,[0,1,2,3,-1])
pyy = np.delete(pyy,3437)
pyt = np.delete(pyt,3437)
'''
'''
##clean-up 2back1
pxx = np.insert(pxx,25,0.705)
pxt = np.insert(pxt,25,-1.132e-8)

pxx = np.delete(pxx,2582)
pxt = np.delete(pxt,2582)

pyy = np.insert(pyy,1,0.660)
pyt = np.insert(pyt,1,1.019e-8)
pyy = np.insert(pyy,25,0.609)
pyt = np.insert(pyt,25,2.0885e-7)

pyy = np.delete(pyy,[2582,2583,5591,5592,5593,5594])
pyt = np.delete(pyt,[2582,2583,5591,5592,5593,5594])
'''

'''
## clean-up back 3 ##
pyy = np.delete(pyy,0)
pyt = np.delete(pyt,0)
pyt = np.insert(pyt,0,1.0463e-7)
pyy = np.insert(pyy,0,0.645)
pyt = np.insert(pyt,29,2.4877e-7)
pyy = np.insert(pyy,29,0.630)
pyt = np.insert(pyt,30,2.5378e-7)
pyy = np.insert(pyy,30,0.663)
pyt = np.insert(pyt,5872,3.031230e-5)
pyy = np.insert(pyy,5872,0.659)

pxx = np.delete(pxx,0)
pxt = np.delete(pxt,0)
pxx = np.insert(pxx,29,0.7)
pxt = np.insert(pxt,29,2.875e-8)
pxx = np.insert(pxx,2681,0.707)
pxt = np.insert(pxt,2681,1.367912e-5)

## clean-up back 2
pyt = np.insert(pyt,952,4.98426e-6)
pyy = np.insert(pyy,952,0.659)

pyt = np.delete(pyt,2577)
pyy = np.delete(pyy,2577)

pxx = np.insert(pxx,5879,0.70)
pxt = np.insert(pxt,5879,3.012198e-5)

## clean-up back 1
pxx = np.insert(pxx,139,0.7132)
pxt = np.insert(pxt,139,5.6517e-7)
pxx = np.insert(pxx,140,0.70)
pxt = np.insert(pxt,140,5.7017e-7)
pxx = np.insert(pxx,141,0.71)
pxt = np.insert(pxt,141,5.7522e-7)
pxx = np.insert(pxx,151,0.689)
pxt = np.insert(pxt,151,6.2492e-7)
pxx = np.insert(pxx,152,0.6996)
pxt = np.insert(pxt,152,6.2982e-7)

#pyy = np.insert(pyy,0,0.6271)
#pyt = np.insert(pyt,0,7.9025e-7)
pyy = np.insert(pyy,139,0.6271)
pyt = np.insert(pyt,139,7.9025e-7)
pyy = np.insert(pyy,150,0.6534)
pyt = np.insert(pyt,150,8.4496e-7)

pyy = pyy[0:6015]
pyt = pyt[0:6015]

pxx = pxx[1:6016]
pxt = pxt[1:6016]

'''

print (len(pxx), len(pyy))

dtx =[]
for i in range (len(pxx)-1):
    d=pxt[i+1] - pxt[i]
    dtx.append(d)
    if d>6e-9 and abs(d-84.48e-9)>1e-9:
        print('bad x peak ',d,i, pxt[i])
dtx = np.asarray(dtx)    

dty =[]
for i in range (len(pyy)-1):
    d= pyt[i+1] - pyt[i]
    dty.append(d)
    if d>6e-9 and abs(d-84.48e-9)>1e-9:
        print('bad y peak ',d,i, pyt[i])
dty = np.asarray(dty)    


dt =[]
for i in range (min(len(pxx), len(pyy))):
    dt.append(pyt[i] - pxt[i])
dt = np.asarray(dt)    

np.savez('%s'%name,pxx=pxx,pxt=pxt,pyy=pyy,pyt=pyt)

plt.title("Beam pickup")
plt.xlabel('T')
plt.ylabel('V')
plt.grid(color='k', linestyle='-', linewidth=2)

plt.plot(t[peaksy[0]-1000:peaksy[-1]+1000], y_normed[peaksy[0]-1000:peaksy[-1]+1000], marker = '', c = 'r',label='M2-4 pickup')
plt.plot(pyt[:], pyy[:], marker='x', linewidth=0, c='r', label='peaks M2-4')
plt.plot(t[peaksx[0]-1000:peaksx[-1]+1000], x_normed[peaksx[0]-1000:peaksx[-1]+1000], marker = '', c = 'b',label='M1-4 pickup')
plt.plot(pxt[:], pxx[:], marker= 'x', linewidth=0, c = 'b',label='peaks M1-4')
#plt.legend()
plt.ylim(-0.2,1.1)
plt.show()

plt.grid(color='k', linestyle='-', linewidth=2)
#plt.plot(pxt[1:],(dtx), c='b')
#plt.plot(pyt[1:],(dty), c='r')
plt.plot(dt/0.8e-10, c='c')
plt.show()
'''
plt.title("Beam pickup (aligned)")
plt.xlabel('T')
plt.ylabel('V')
plt.grid(color='k', linestyle='-', linewidth=2)
shift = int(np.median(dt/1.6e-10))
print(shift,peaksy[3]-peaksy[2])
plt.plot(t[peaksx[0]:peaksx[100]], y_normed[peaksy[0]:peaksy[100]], marker = '', c = 'r',label='M2-4')
plt.plot(pxt[0:100], pxx[0:100], marker= 'x', linewidth=0, c = 'b',label='peaks M1-4')
plt.plot(t[peaksx[0]:peaksx[100]], x_normed[peaksx[0]:peaksx[100]], marker = '',linewidth=1, linestyle=':',c = 'b',label='M1-4')
plt.show()


delta = []
for i in range (peaksx[0],peaksx[0]+1000):
    delta.append( -x[i] + y[i-172])

plt.plot(t[peaksx[0]:peaksx[0]+1000],y[peaksx[0]-172:peaksx[0]+1000-172], c='b')
plt.plot(t[peaksx[0]:peaksx[0]+1000],x[peaksx[0]:peaksx[0]+1000], c='r')
plt.plot(t[peaksx[0]:peaksx[0]+1000], delta[:], c='g')
plt.plot(pxt[:100],pxx[:100], marker='x',linewidth=0,  c='r')
plt.plot(pxt[:100],pyy[:100], marker='x',linewidth=0,  c='b')
plt.show()

plt.scatter(delta[:],x[peaksx[0]:peaksx[0]+1000], c='m')
plt.scatter(delta[:],y[peaksx[0]-172:peaksx[0]+1000-172], c='c')
plt.scatter(pxx,pyy, c='y')
plt.ylim(-0.1,0.4)
plt.show()
'''
