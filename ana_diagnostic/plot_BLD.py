import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.spatial import distance
import sklearn.mixture

plt.rcParams["font.family"] = "Times New Roman"

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_fwhm(arrayx, arrayy):
    peaks, _ = find_peaks(arrayy, height=0.)
    max_p = np.argmax([arrayy[int(xx)] for xx in peaks])
    
    fwhm = peak_widths(arrayy, peaks, rel_height=0.5)
        
    mid_x = arrayx[int(peaks[max_p])]
    mid_y = arrayy[int(peaks[max_p])]
    
    min_x, max_x = arrayx[int(fwhm[2][max_p])], arrayx[int(fwhm[3][max_p])]
    #min_y, max_y = arrayy[int(fwhm[2][max_p])], arrayy[int(fwhm[3][max_p])]

    return mid_y, fwhm[1][max_p], mid_x, min_x, max_x

def gauss(x):
    gmm = sklearn.mixture.GaussianMixture()
    r = gmm.fit(x[:, np.newaxis]) # GMM requires 2D data as of sklearn version 0.16

    print("mean : %f, var : %f" % (r.means_[0, 0], r.covariances_[0, 0]))

    xlin = np.linspace(np.min(x),np.max(x),1000).reshape(-1,1)
    pdf = np.exp(gmm.score_samples(xlin))
    
    return r,pdf

########################################
x = []
y = []
'''
files = ["BLD_devices_From_2021-10-19+19:47:00_to_2021-10-19+19:47:45.csv", #-3 not optimized
         "BLD_devices_From_2021-10-19+19:43:00_to_2021-10-19+19:44:00.csv", #-2 not optimized
         "BLD_devices_From_2021-10-19+19:40:00_to_2021-10-19+19:41:00.csv", #-1 not optimized
         "BLD_devices_From_2021-10-19+21:49:20_to_2021-10-19+21:53:20.csv", #-1 optimized
         "BLD_devices_From_2021-10-19+21:19:30_to_2021-10-19+21:22:00.csv", #-2 optimized
         "BLD_devices_From_2021-10-19+20:44:30_to_2021-10-19+20:46:30.csv", #-3 optimized
         "BLD_devices_From_2021-10-19+18:49:30_to_2021-10-19+18:50:30.csv", #+0.15 
         "BLD_devices_From_2021-10-19+19:02:30_to_2021-10-19+19:04:00.csv", #-0.15
         "BLD_devices_From_2021-10-19+19:35:00_to_2021-10-19+19:36:00.csv", #+1 not optimized
         "BLD_devices_From_2021-10-19+19:31:00_to_2021-10-19+19:32:00.csv", #+2 not optimized
         "BLD_devices_From_2021-10-19+19:14:30_to_2021-10-19+19:17:00.csv", #+3 not optimized
         "BLD_devices_From_2021-10-25+15:52:10_to_2021-10-25+15:53:10.csv", #+1 optimized
         "BLD_devices_From_2021-10-25+09:42:00_to_2021-10-25+09:42:40.csv", #+2 optimized
         "BLD_devices_From_2021-10-25+09:09:40_to_2021-10-25+09:10:40.csv",
         "BLD_devices_From_2021-11-02+08:25:00_to_2021-11-02+08:26:00.csv"] #+3 optimized
'''
files = ['15062022_bld.csv']
pos = [-3, -2, -1, -1, -2, -3, -0.15, 0.15, 1, 2, 3, 1, 2, 3, 3]
cols = ['b','r','g','y','m','c','k','k','g','r','b','g','r','b','b']
mark = ['o','o','o','x','x','x','s','s','^','^','^','x','x','x','x']
ls = ['-','-','-','-','-','-',':','-','--','--','--','-','-','-',':']
add = ['10/19','10/19','10/19','10/19 opt','10/19 opt.','10/19 opt','10/19','10/19','10/19','10/19','10/19','10/25 opt','10/25 opt','10/25 opt','11/02']

plt.title("BLD phase scans")
plt.xlabel('phase (deg)')
plt.ylabel('signal (V)')
#plt.ylabel('wire position (mm)')
#plt.xlim(15.,45.)  
plt.grid(which='both',color='k', linestyle='-', linewidth=2)

for i in range(len(files)):
    x.append([])
    y.append([])
    #files.append(("scan_ddmot3_wire%smm_1.csv")%str(pos[i]))

txtblock = 'FWHM:'
for j in range(8,len(files)):
    files[j] = '/Users/rshara01/WORK/BLD/data/%s'%files[j]
    for line in open(files[j], 'r'):
        lines = [i for i in line.split(',')]
        x[j].append(float(lines[5]))
        y[j].append(float(lines[3]))
    #plt.plot(x[j],y[j], c=cols[j], linestyle=ls[j], label=('wire %smm %s')%(str(pos[j]),add[j]) )
    plt.plot(x[j], y[j], marker = mark[j] , c = cols[j], label = ('wire %smm')%str(pos[j]))
    mid_y, h, mid_x, min_x, max_x = find_fwhm(x[j],y[j])
    plt.plot(mid_x, mid_y, "x")
    plt.hlines(h, min_x, max_x, color=cols[j], linestyle=ls[j])
    ps = 1./201.5e6 * np.abs(max_x-min_x)/360. * 1e12
    #print(mid_x,ps)
    txtblock = txtblock + ('\n%smm: %.2fps')%(str(pos[j]),ps)


print(txtblock)    
#plt.text(15.0, 3., 'matplotlib', horizontalalignment='left',verticalalignment='center')
'''
plt.text(15.0, 2., txtblock, size=10, rotation=0.,
         ha="left", va="bottom",multialignment='center',
         bbox=dict(boxstyle="square",
                   ec=(0., 0.5, 0.5),
                   fc=(0., 0.8, 0.8),
                   )
         )
'''
plt.legend()
plt.show()

