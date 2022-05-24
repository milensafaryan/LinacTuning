import os,sys,platform
import functools
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
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
                         

def plot_one_avg(df,which):
    colnames = [col for col in list(df.keys()) if col.find('Time')==-1]    
    try:
        x_m = [np.median(df[col]) for col in colnames if len(df[col])>0]
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


def filter_and_plot_multiFile(files,thresh,m,which):
    plt.rc("figure",figsize=(20,12))
    overlap = {name for name in mcolors.CSS4_COLORS
               if f'xkcd:{name}' in mcolors.XKCD_COLORS}

    overlap.difference_update(['aqua','ivory','white','lime','chocolate','gold'])
    colors = [mcolors.XKCD_COLORS[f'xkcd:{color_name}'].upper() for color_name in sorted(overlap)]    
    labels = get_labels(files)

    listdf = [None]*len(files)
    
    for i in range(len(files)):
        listdf[i] = read_data(files[i],which)

    colnames = [col for col in list(listdf[0].keys()) if col.find('Time')==-1] # reference

    numcol = int(len(listdf)/2) if len(listdf)%2==0 else int(len(listdf)/2)+1
    fig, axs = plt.subplots(numcol,2,sharex=True, sharey=True)

    print(len(listdf), len(listdf)/2)

    for k,df in enumerate(listdf):
        # remove outlier samples
        #reject_outliers(df,m)
        # remove noisy
        filter_noisy(df,thresh,False)

        delta = []
        idx = []
        if k!=0:
            for l,col in enumerate(colnames):
                if col in list(df.keys()) and col in list(listdf[0].keys()):
                    if len(df[col])>0 and len(listdf[0][col])>0:
                        delta.append(np.median(df[col]) - np.median(listdf[0][col]))
                        idx.append(l)
            
            axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].fill_between(idx,delta,facecolor=colors[k],label='%s'%labels[k])
            axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].legend(loc='upper left', fancybox=True, fontsize='small')
            axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].xaxis.set_tick_params(direction='in', which='major')
            if k==(numcol-1) or (k==(numcol*2-1) and numcol==int(len(listdf)/2)) or (k==(numcol*2-2) and numcol==int(len(listdf)/2+1)):
                axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].set_xticks(np.arange(len(colnames)),colnames, rotation = 'vertical')
            axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].grid(True)
            
            if which=='phase':
                axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].set_ylim(-3,3)
                fig.supylabel(' Delta Phase (deg)')
                fig.suptitle('BPM phases',fontsize=12)
            elif which=='bpv':    
                axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].set_ylim(-1.,1)
                fig.supylabel('Delta Y (mm)')    
                fig.suptitle("BPM Vertical positions")
            elif which=='bph':
                axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].set_ylim(-1.,1)
                fig.supylabel('Delta X (mm)')    
                fig.suptitle("BPM Horizontal positions")
            elif which=='blm':
                axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].set_ylim(-3,3)
                fig.supylabel('Delta Loss (cnt)')
                fig.suptitle('BLMs')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(left=0.06)
    plt.subplots_adjust(right=0.98)
    #plt.show()

    plt.savefig('%s_%s_%s.png'%(labels[-1].split(' ')[0].replace('/','-'),labels[-1].split(' ')[1].replace(':','-'),which))

def get_files(path,date):
    files =[]
    datestr= [date.strftime('%Y_%m_%d-%H')]
    for i in range(1,24):
        date = date - timedelta(hours=1)
        datestr.append(date.strftime('%Y_%m_%d-%H'))

    for subdir, dirs, fls in os.walk(path):
        for file in fls:
            for ds in datestr:
                if ds in file:
                    files.append(os.path.join(subdir,file))
    files.sort()
    return files

def get_labels(files):
    labels = []
    for f in files:
        datestr = f.split('/')[-1].split('-')[0].replace('_','/')
        labels.append('%s %s:%s'%(datestr,f.split('-')[-1].split('_')[0],f.split('-')[-1].split('_')[1]))

    return labels

def check_col_order(files,which):
    lists =[get_colnames(f,which) for f in files]

    for i in range(len(lists)-1):
        if functools.reduce(lambda x, y: x and y, map(lambda a, b: a == b, lists[i], lists[i+1]), True):
            continue
        else:
            print("Column names not in the same order! Check input data.")


def parse_args():

    parser = argparse.ArgumentParser(description="usage: %prog [options] \n")
    parser.add_argument ('--i',  dest='datadir', default='/accelai/data/rshara01',
                         help="Input data directory path")
    parser.add_argument ('--d', dest='date', default='',
                         help="End date/time to plot. Format %Y_%m_%d-%H")
    #parser.add_argument ('--o',  dest='imgdir', default='',
    #                     help="Image save directory")

    options = parser.parse_args()
    datadir = options.datadir
    datestr = options.date
    #imgdir  = options.imgdir

    if datestr =='':
        date = datetime.today()
    else:
        try:
            date = datetime.strptime(datestr,'%Y_%m_%d-%H_%M_%S')
        except:
            print('Invalid date format. Using default.')
            date = datetime.today()

    return datadir,date

def main():
    
    path,date = parse_args()
    files = get_files(path,date)

    filter_and_plot_multiFile(files,100,3,'phase')
    filter_and_plot_multiFile(files,10,3,'blm')
    filter_and_plot_multiFile(files,10,3,'bph')
    filter_and_plot_multiFile(files,10,3,'bpv')

    #filter_and_plot_singleFile(files[4],1,3,'bph')
    

if __name__ == "__main__":
    main()
