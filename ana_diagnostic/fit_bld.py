import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import sklearn.mixture
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["savefig.format"] = 'pdf'

'''
def gauss(x):
    gmm = sklearn.mixture.GaussianMixture()
    # if x is 1-D:
    r = gmm.fit(x[:, np.newaxis])
    #r = gmm.fit(x)
    print("mean : %f, var : %f" % (r.means_[0, 0], r.covariances_[0, 0]))

    xlin = np.linspace(np.min(x),np.max(x),1000).reshape(-1,1)
    pdf = np.exp(gmm.score_samples(xlin))

    return r,pdf
'''

def gauss(x, H, A, x0, sigma):
    #return 0.75 + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y, e):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma], sigma=e, absolute_sigma=True)

    return popt,pcov

    
def peakfinder(arrayy):
    peaks, _ = find_peaks(arrayy, height=4.5, distance=100, prominence = 0.6)

    widths = peak_widths(arrayy, peaks, rel_height=0.93)
    
    return peaks, widths

def plot_peaks_timedomain(x,y,peaks,widths):
        mid_x = [x[idx] for idx in peaks]
        mid_y = [y[idx] for idx in peaks]
        min_x = [x[round(idx,0)] for idx in widths[2]]
        max_x = [x[round(idx,0)] for idx in widths[3]]
        min_y = [y[round(idx,0)] for idx in widths[2]]
        max_y = [y[round(idx,0)] for idx in widths[3]]
        #print(peaks,widths )
        
        plt.plot(x, y, marker = '.' , c = 'r', label = ('wire %smm')%str(0))
        plt.plot(mid_x, mid_y, "x")
        plt.plot(min_x,min_y,'o')
        plt.plot(max_x,max_y,'p')
        plt.hlines(widths[1],min_x, max_x, linestyle='--')

        plt.show()

def plot_lin_fit(x,y,yerr,xlabel,ylabel,ref,fit_flat=False):
        plt.errorbar(x,y,yerr=yerr,fmt='o',color='r',label='syst. error')
        plt.xlabel('%s'%xlabel, fontsize=18)
        plt.ylabel('%s'%ylabel, fontsize=18)
        plt.xticks(fontsize=16 )
        plt.yticks(fontsize=16 )

        w = [1/s for s in yerr]
        plin,res,_,_,_ = np.polyfit(x,y,deg=1,w=w,full=True)

        p = np.poly1d(plin)
        dof = len(x)-(1+1)
        textstr = r'$\chi^2/DoF=%.2f/%d$' % (res, dof,)
        plt.plot(x,p(x),c='k',label='linear %s'%textstr)
        plt.scatter(x[ref],y[ref],s=80, facecolors='none', edgecolors='b',label='reference')
        
        if fit_flat:
            pflat,resf,_,_,_ = np.polyfit(x,y,deg=0,w=w,full=True)
            doff = len(x)-1
            textstr = r'$\chi^2/DoF=%.2f/%d$' % (resf, doff,)
            plt.plot(x,pflat*np.ones(len(x)),'g--',label='flat %s'%textstr)


        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(top=0.92)
        plt.subplots_adjust(left=0.12)
        plt.subplots_adjust(right=0.96)

        plt.legend(fontsize=16)
        plt.show()


def get_mean_stat_syst(data,err):
    mean = np.mean(data)
    stat = np.sqrt(np.sum(err**2))
    syst = np.max(data)-np.min(data)

    return mean, stat, syst
        
def main():
    path = '/Users/rshara01/WORK/BLD/data/'
    #files = ['15062022_bld.csv','BLD_devices_From_2021-12-15+13:00:00_to_2021-12-15+15:30:00.csv']
    #files = ['BLD_devices_From_2021-12-15+13:00:00_to_2021-12-15+15:30:00.csv']
    files = ['BLD_devices_From_2022-01-21+08:00:00_to_2022-01-21+11:00:00.csv']
    dfs = []
    
    for f in files:
        f = '%s%s'%(path,f)
        #dfs.append(pd.read_csv(f))
        df = pd.read_csv(f)

        df.columns = ['TS L:D03BDM','L:D03BDM','TS L:D03BDS','L:D03BDS','TS L:DDMOT3','L:DDMOT3','TS L:DDMOT4','L:DDMOT4','TS L:D03HV1','L:D03HV1','TS L:D03HV2','L:D03HV2']

        #df = df.loc[df['TS L:D03BDM']>1639601700]
        #df = df.loc[df['TS L:D03BDM']>1639602220]
        df = df.loc[df['TS L:D03BDM']>1642780800]
        df = df.loc[df['TS L:D03BDM']<1642782120]
        df = df.loc[df['L:DDMOT3']>0]
        df.reset_index(inplace=True, drop=True)
        
        
        x = df.iloc[:,0] #TS
        y = df.loc[:,'L:D03BDS']
        z = df.loc[:,'L:DDMOT3']
        w = df.loc[:,'L:D03BDM']
        
        peaks, widths = peakfinder(y)
        fitranges = [z for z in zip(widths[2],widths[3])]
        #plot_peaks_timedomain(x,y,peaks,widths)
        #plot_peaks_timedomain(z,y,peaks,widths)

        # get error on ydata from jitter on first 200 samples
        rms = np.std(y[:200])
        
        xdata = []
        ydata = []
        
        for f in fitranges:
            xdata.append(z[int(f[0]):int(f[1])])
            ydata.append(y[int(f[0]):int(f[1])])

        # mean vs T5 phase
        x0 = np.zeros((5,2),dtype=float)
        x0err = np.zeros((5,2),dtype=float)
        sigma = np.zeros((5,2),dtype=float)
        sigmaerr = np.zeros((5,2),dtype=float)
        
        means1 = np.zeros(5)
        staterr1 = np.zeros(5)
        systerr1 = np.zeros(5)
        means2 = np.zeros(5)
        staterr2 = np.zeros(5)
        systerr2 = np.zeros(5)

        t5 = [-32.9,-32.4,-31.9,-30.9,-31.4]

        # fig, ax = plt.subplots(1,5,figsize=(10,4), sharey=True)
        fig, ax = plt.subplots()
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        title = ['T5$-1.0^\circ$','T5$-0.5^\circ$','T5 REFERENCE','T5$+1.0^\circ$','T5$+0.5^\circ$']
        colors = ['r','orange']
        markers = ['ko','bx']

        
        for j in range(5):
            for i in range(2):
                errors = np.ones(len(ydata[j*2+i]))*rms
                (H, A, x0[j][i], sigma[j][i]),pcov = gauss_fit(xdata[j*2+i], ydata[j*2+i], errors)
                FWHM = 2.35482 * sigma[j][i]
                ps = 1./201.5e6 * FWHM/360. * 1e12

                x0err[j][i] = np.sqrt(pcov[2][2])
                sigmaerr[j][i] = np.sqrt(pcov[3][3])
                
                textstr = '\n'.join((
                    r'$\mu=%.2f\pm%.2f$' % (x0[j][i], np.sqrt(pcov[2][2]),),
                    r'$\sigma=%.2f\pm%.2f$' % (sigma[j][i], np.sqrt(pcov[3][3]),),
                    r'$\mathrm{FWHM}=%.2f\mathrm{ps}$' % (ps, )))
                '''
                ax[j].plot(xdata[j*2+i], ydata[j*2+i], markers[i], label='data %d'%i)
                ax[j].plot(xdata[j*2+i], gauss(xdata[j*2+i], *gauss_fit(xdata[j*2+i], ydata[j*2+i], errors)[0]), '-',color=colors[i], label='fit %d'%i)
                
                ax[j].text(0.05, 0.95-i*0.15, textstr, transform=ax[j].transAxes, color=colors[i], fontsize='small',
                           verticalalignment='top', bbox=props)
                '''
                if j==2:
                    ax.plot(xdata[j*2+i], ydata[j*2+i], markers[i], label='data %d'%i)
                    ax.plot(xdata[j*2+i], gauss(xdata[j*2+i], *gauss_fit(xdata[j*2+i], ydata[j*2+i], errors)[0]), '-',color=colors[i], label='fit %d'%i)
                    ax.text(0.05, 0.95-i*0.25, textstr, transform=ax.transAxes, color=colors[i], fontsize=15,
                            verticalalignment='top', bbox=props)
                    
            means1[j],staterr1[j],systerr1[j] = get_mean_stat_syst(x0[j],x0err[j])
            means2[j],staterr2[j],systerr2[j] = get_mean_stat_syst(sigma[j],sigmaerr[j])
                          
            #ax[j].set_title('%s'%title[j])
            ax.set_title('Tank 5 phase =%.1f degrees'%t5[2], fontsize=18)
            ax.tick_params(axis='both',labelsize=16)
        

        fig.supylabel('EMT signal (V)', fontsize=18)
        fig.suptitle('Degrees at 201 MHz', y=0.05, fontsize=18)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(top=0.92)
        plt.subplots_adjust(left=0.10)
        plt.subplots_adjust(right=0.98)
        plt.show()
        
        plot_lin_fit(t5,means2,systerr2,'Tank 5 phase (degrees)','$1\sigma$ Bunch Length (deg @ 201 MHz)',2,fit_flat=True)
        plot_lin_fit(t5,means1,systerr1,'Tank 5 phase (degrees)','BLD Signal Mean (deg @ 201 MHz)',2)

        
if __name__ == '__main__':
    main()
