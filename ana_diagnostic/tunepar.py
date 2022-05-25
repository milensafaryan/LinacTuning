import numpy as np
from scipy.signal import find_peaks
import pylab as plt
import matplotlib.transforms as mtransforms
from matplotlib.widgets import RangeSlider,Slider, Button

def zoom_event(event):
    global fill
    trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
    if (fill in ax2.collections):
        ax2.collections.remove(fill)
    fill=ax2.fill_between(t,0,1,where=np.logical_and(t>ax1.get_xlim()[0],t<ax1.get_xlim()[1]),facecolor='green', alpha=0.5,transform=trans)

def update(event):
    (pxt,pxx,dtx)=peak_finder(x_norm,t,heightx_slider.val,prom_slider.val,dist_slider.val)
    (pyt,pyy,dty)=peak_finder(y_norm,t,heighty_slider.val,prom_slider.val,dist_slider.val)

    old1_xlim=ax1.get_xlim()
    old1_ylim=ax1.get_ylim()
    old2_xlim=ax2.get_xlim()
    old2_ylim=ax2.get_ylim()
    ax1.cla()
    ax1.set_title("Npeaks(x)=%i; Npeaks(y)=%i"%(len(pxt),len(pyt)))
    ax1.plot(t,x_norm)
    ax1.plot(pxt,pxx,marker='o',linewidth=0)
    ax1.plot(t,y_norm)
    ax1.plot(pyt,pyy,marker='o',linewidth=0)
    ax2.cla()
    ax2.plot(pxt[1:],(dtx), c='b')
    ax2.plot(pyt[1:],(dty), c='r')
    
    if (event is None):
        ax1.set_xlim(1.3e-5,1.34e-5)
        ax2.set_ylim(0,1.5e-7)
    else:
        ax1.set_xlim(old1_xlim)
        ax1.set_ylim(old1_ylim)
        ax2.set_xlim(old2_xlim)
        ax2.set_ylim(old2_ylim)
        fig.canvas.draw_idle()
    zoom_event(None)
        
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

def reset(event):
    ax1.set_xlim(ax2.get_xlim())
    ax1.set_ylim(0,1)
    #ax2.set_xlim(t.min(0),t.max(0))
    fig.canvas.draw_idle()


init_heightx=(0.3,1)
init_heighty=(0.2,1)
init_dist=60
init_prom=(0.2,1)

fill=None
t = []
x = [] 
y = [] 
z = []

path = r'/Users/rshara01-local/Desktop/Winter 2022/Study 12302021/2study12302021/'
suffix = '_ALL.csv'
filename = path + r'2back1' + suffix

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
#x[x<0]=0
#y[y<0]=0
#x=-x
#y=-y
x_norm=(x-x.min(0))/x.ptp(0)
y_norm=(y-y.min(0))/y.ptp(0)

(pxt,pxx,dtx)=peak_finder(x_norm,t,init_heightx,init_prom,init_dist)
(pyt,pyy,dty)=peak_finder(y_norm,t,init_heighty,init_prom,init_dist)

fig, (ax1, ax2)=plt.subplots(2,1,figsize=(12,6))
plt.subplots_adjust(left=0.05,right=0.55)

axheightx = plt.axes([0.65, 0.8, 0.25, 0.03])
heightx_slider = RangeSlider(
    ax=axheightx,
    label="Height sig 1",
    valmin=0,
    valmax=1,
    valinit=init_heightx,
    orientation="horizontal"
)

axheighty = plt.axes([0.65, 0.6, 0.25, 0.03])
heighty_slider = RangeSlider(
    ax=axheighty,
    label="Height sig 2",
    valmin=0,
    valmax=1,
    valinit=init_heighty,
    orientation="horizontal"
)

axprom = plt.axes([0.65, 0.4, 0.25, 0.03])
prom_slider = RangeSlider(
    ax=axprom,
    label="Prominence",
    valmin=0,
    valmax=1,
    valinit=init_prom,
    orientation="horizontal"
)
axdist = plt.axes([0.65, 0.2, 0.25, 0.03])
dist_slider = Slider(
    ax=axdist,
    label="Distance",
    valmin=1,
    valmax=100,
    valinit=init_dist,
    orientation="horizontal"
)

heightx_slider.on_changed(update)
heighty_slider.on_changed(update)
prom_slider.on_changed(update)
dist_slider.on_changed(update)

update(None)

axreset = plt.axes([0.05, 0.9, 0.1, 0.04])
button = Button(axreset, 'Reset zoom', hovercolor='0.975')

button.on_clicked(reset)

fig.canvas.mpl_connect('motion_notify_event',zoom_event)
fig.canvas.mpl_connect('button_release_event',zoom_event)
zoom_event(None)

def move_zoom(event):
    if event.inaxes==ax2:
        print(event.xdata)
        new_limit=(event.xdata,event.xdata+ax1.get_xlim()[1]-ax1.get_xlim()[0])
        ax1.set_xlim(new_limit)
        zoom_event(event)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event',move_zoom)

plt.show()
