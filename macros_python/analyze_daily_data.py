import io
import os,sys,platform
import numpy as np
import matplotlib.pyplot as plt
import functools
from datetime import datetime
import pandas as pd

def read_data(filename,which):
    dataset = pd.read_csv('%s'%filename)
    df = dataset.copy()
    if which=='phase':
        df =df.filter(regex = 'Time|F', axis=1)
        df.drop(list(df.filter(regex='DRQF|SSDABF|PAH|D04|BP20')),axis=1,inplace=True)
    elif which=='blm':
        df = df.filter(regex = 'Time|LM', axis=1)
        df.drop(list(df.filter(regex='LMSM')),axis=1,inplace=True)
    elif which=='bph':
        df = df.filter(regex = 'Time|BPH|HP', axis=1)
        df.drop(list(df.filter(regex='D04|BPH20')),axis=1,inplace=True)
    elif which=='bpv':
        df = df.filter(regex = 'Time|BPV|VP', axis=1)
        df.drop(list(df.filter(regex='D04|BPV20|PHAS')),axis=1,inplace=True)

    df['Time'] = pd.to_datetime(df['Time'])
    df.dropna(inplace=True)

    return df


def filter_noisy(df,threshold,verbose):
    cols = [ col for col in list(df.keys()) if col.find('Time')==-1]
    dropcol = [col for col in cols if np.var(df[col])>threshold or np.var(df[col])==0]
    if verbose:
        [ print(np.var(df[cols[i]])) for i in range(len(cols)) ]
        print('idx: ',idx)

    df.drop(columns=dropcol,inplace=True)
    
    return None

def reject_outliers(df,m=3):
    cols = [ col for col in list(df.keys()) if col.find('Time')==-1]
    df_sub = df.loc[:,cols]
    iqr = df_sub.quantile(0.75, numeric_only=False) - df_sub.quantile(0.25, numeric_only=False)
    lim = np.abs((df_sub - df_sub.median()) / iqr) < 2.22
    
    df.loc[:, cols] = df_sub.where(lim, np.nan)
    df.dropna(subset=cols, inplace=True)

    
def calc_delta(xref,x,idx):
    xref_m = [np.mean(xref[i]) for i in idx]
    x_m = [np.mean(x[i]) for i in idx]

    dx = np.subtract(x_m,xref_m)
    return dx
    
def plot_raw(df,start,stop,which):
    plt.grid(color='k', linestyle='-', linewidth=1.)
    plt.xlabel('Timestamp')

    if which=='phase':
        plt.ylim(-185,185)
        plt.ylabel('Phase (deg)')
        plt.title('BPM phases')
    elif which=='bpv':    
        plt.ylim(-10.,10)
        plt.ylabel('Y (mm)')    
        plt.title("BPM Vertical positions")
    elif which=='bph':
        plt.ylim(-10.,10)
        plt.ylabel('X (mm)')    
        plt.title("BPM Horizontal positions")
    elif which=='blm':
        plt.ylim(-1,5)
        plt.ylabel('Loss (cnt)')
        plt.title('BLMs')

    colnames = [ col for col in list(df.keys()) if col.find('Time')==-1]
    [ plt.plot(df['Time'],df[colnames[i]],label='%s'%colnames[i]) for i in range(int(start),int(stop))]
    plt.legend()
    plt.show()

def plot_phases_norm(df,start,stop):
    plt.grid(color='k', linestyle='-', linewidth=1.)
    plt.ylim(-0.05,8.05)
    plt.xlabel('Timestamp')
    plt.ylabel('Phase (normalized)')
    plt.title("BPM Phases")


    colnames = [ col for col in list(df.keys()) if col.find('Time')==-1]
    for i in range(int(start),int(stop)):
        xnorm = np.asarray(df[colnames[i]])
        try:
            #xnorm = (xnorm - xnorm.min(0)) / xnorm.ptp(0)
            xnorm = (xnorm - xnorm.min(0))
        except ValueError:
            pass
        
        plt.plot(df['Time'],xnorm,label='%s'%colnames[i])

    plt.legend()
    plt.show()
                         

def plot_all_dx(idxm,dxm,idxp,dxp,colnames):
    fig, axs = plt.subplots(7,1,sharex=True)
    fig.set_size_inches(12.,8.)
    fig.supylabel(r'$\Delta')
                
    labels = ['RFQ','RFB','Tank1','Tank2','Tank3','Tank4','Tank5']
    for i in range(7):
        axs[i].plot(idxm[i],dxm[i],label='%s Minus 1 deg'%labels[i])
        axs[i].plot(idxp[i],dxp[i],label='%s Plus 1 deg'%labels[i])
        pxs[i].legend(loc='upper right', fancybox=True, fontsize='small')

    for i,ax in enumerate(axs.flat):
        ax.set_title(labels[i], y=1.0, pad=-14)
        ax.label_outer()

    ticks = [i for i in range(len(colnames))]
    plt.xticks(ticks,colnames, rotation = 'vertical')
    plt.subplots_adjust(bottom=0.13)
    plt.subplots_adjust(top=0.94)
    plt.subplots_adjust(left=0.1)
    plt.subplots_adjust(right=0.95)

    plt.show()


def plot_one_avg(df,which):
    colnames = [col for col in list(df.keys()) if col.find('Time')==-1]    
    try:
        x_m = [np.mean(df[col]) for col in colnames if len(df[col])>0]
        idx = [j for j,col in enumerate(colnames) if len(df[col])>0 ]
    except ValueError:
        pass

    plt.grid(color='k', linestyle='-', linewidth=1.)
    plt.plot(idx,x_m,marker = '', c = 'g',label='')

    if which=='phase':
        plt.ylim(-185,185)
        plt.ylabel('Phase (deg)')
        plt.title('BPM phases')
    elif which=='bpv':    
        plt.ylim(-10.,10)
        plt.ylabel('Y (mm)')    
        plt.title("BPM Vertical positions")
    elif which=='bph':
        plt.ylim(-10.,10)
        plt.ylabel('X (mm)')    
        plt.title("BPM Horizontal positions")
    elif which=='blm':
        plt.ylim(-1,5)
        plt.ylabel('Loss (cnt)')
        plt.title('BLMs')


    plt.xticks(np.arange(len(colnames)),colnames, rotation = 'vertical')
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(top=0.94)
    plt.subplots_adjust(left=0.11)
    plt.subplots_adjust(right=0.95)

    plt.show()
    
def filter_and_plot_singleFile(filename,thresh,m,which):
    plt.rc("figure",figsize=(10,6))
    plt.rc("axes",prop_cycle= plt.cycler("color", plt.cm.tab20.colors))
    
    df = read_data(filename,which)
    # remove noisy devices
    filter_noisy(df,thresh,False)
    # remove outlier samples from devices
    reject_outliers(df,m)
    
    # plot raw BPM phase data
    x = (len(df.keys())-1)
    for i in range(int(x/12.)+1):
        if which=='phase':
            plot_phases_norm(df,i*12,min(x,(i+1)*12))
        else:
            plot_raw(df,i*12,min(x,(i+1)*12),which)
    
    #plot averages
    plot_one_avg(df,which)


def filter_and_plot_multiFile(files,thresh,m,which,REF):
    plt.rc("figure",figsize=(10,6))
    plt.rc("axes",prop_cycle= plt.cycler("color", plt.cm.tab20.colors))
    
    labels = get_labels(files)

    kk = int([i for i,s in enumerate(labels) if REF in s][0])
            
    listdf = [None]*len(files)
    
    for i in range(len(files)):
        listdf[i] = read_data(files[i],which)
    colnames = [col for col in list(listdf[kk].keys()) if col.find('Time')==-1] # reference
        
    for k,df in enumerate(listdf):
        # remove noisy
        filter_noisy(df,thresh,False)
        # remove outlier samples
        reject_outliers(df,m)
        print('df%d: '%k,df.keys())
        
        delta = []
        idx = []
        if k!=kk:
            for l,col in enumerate(colnames):
                if col in list(df.keys()):
                    delta.append(np.median(df[col]) - np.median(listdf[kk][col]))
                    idx.append(l)
            
            plt.plot(idx,delta,marker = '.',linestyle='None',label='%s'%labels[k])

                
    plt.grid(color='k', linestyle='-', linewidth=1.)

    if which=='phase':
        plt.ylim(-10,10)
        plt.ylabel('Phase (deg)')
        plt.title('BPM phases')
    elif which=='bpv':    
        plt.ylim(-10.,10)
        plt.ylabel('Y (mm)')    
        plt.title("BPM Vertical positions")
    elif which=='bph':
        plt.ylim(-10.,10)
        plt.ylabel('X (mm)')    
        plt.title("BPM Horizontal positions")
    elif which=='blm':
        plt.ylim(-10,10)
        plt.ylabel('Loss (cnt)')
        plt.title('BLMs')
    
    plt.xticks(np.arange(len(colnames)),colnames, rotation = 'vertical')
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(top=0.94)
    plt.subplots_adjust(left=0.11)
    plt.subplots_adjust(right=0.95)
    plt.legend(loc='best',ncol = 4, fancybox=True, fontsize='small')
    plt.show()


def get_files(path,date):
    files =[]
    datestr = date.strftime('%Y_%m_%d')

    for subdir, dirs, fls in os.walk(path):
        for file in fls:
            if datestr in file:
                files.append(os.path.join(subdir,file))
    files.sort()
    return files[:6]

def get_labels(files):
    labels = []
    for f in files:
        labels.append('%s:%s'%(f.split('-')[-1].split('_')[0],f.split('-')[-1].split('_')[1]))

    return labels[:6]

def check_col_order(files,which):
    lists =[get_colnames(f,which) for f in files]

    for i in range(len(lists)-1):
        if functools.reduce(lambda x, y: x and y, map(lambda a, b: a == b, lists[i], lists[i+1]), True):
            continue
        else:
            print("Column names not in the same order! Check input data.")

def main():

    path = r'/Users/rshara01-local/Desktop/LINAC_STUDY/Daily_snapshots'
    date = datetime.strptime('2022-05-18','%Y-%m-%d')

    files = get_files(path,date)

    filter_and_plot_multiFile(files,1,3,'phase','00:00')
    #filter_and_plot_multiFile(files,1,2.5,'blm','07MAR2022')
    #filter_and_plot_multiFile(files,1,2.5,'bph','07MAR2022')
    #filter_and_plot_multiFile(files,1,2.5,'bpv','07MAR2022')

    #filter_and_plot_singleFile(files[4],1,3,'bph')
    #filter_and_plot_singleFile(files[4],1,3,'bpv')
    #filter_and_plot_singleFile(files[0],100,3,'phase')
    #filter_and_plot_singleFile(files[1],100,3,'phase')
    #filter_and_plot_singleFile(files[4],100,3,'phase')
    #filter_and_plot_singleFile(files[20],10,3,'blm')
    

if __name__ == "__main__":
    main()
