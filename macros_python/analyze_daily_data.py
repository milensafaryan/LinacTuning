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
        df.drop(list(df.filter(regex='DRQF|SSDABF|PAH|D04|BP20|B:')),axis=1,inplace=True)
    elif which=='blm':
        df = df.filter(regex = 'Time|LM', axis=1)
        df.drop(list(df.filter(regex='LMSM|B:')),axis=1,inplace=True)
    elif which=='bph':
        df = df.filter(regex = 'Time|BPH|HP', axis=1)
        df.drop(list(df.filter(regex='D04|BPH20|B:')),axis=1,inplace=True)
    elif which=='bpv':
        df = df.filter(regex = 'Time|BPV|VP', axis=1)
        df.drop(list(df.filter(regex='D04|BPV20|PHAS|B:')),axis=1,inplace=True)

    df['Time'] = pd.to_datetime(df['Time'])
    df.dropna(inplace=True)

    return df


def filter_noisy(df,threshold,verbose):
    cols = [ col for col in list(df.keys()) if col.find('Time')==-1]
    dropcol = [col for col in cols if np.std(df[col])>threshold or np.var(df[col])==0]
    if verbose:
        [ print(col,np.std(df[col])) for col in dropcol ]

    df.drop(columns=dropcol,inplace=True)
    
    return None

def reject_outliers(df,m=3):
    cols = [ col for col in list(df.keys()) if col.find('Time')==-1]
    df_sub = df.loc[:,cols]
    '''
    iqr = df_sub.quantile(0.75, numeric_only=False) - df_sub.quantile(0.25, numeric_only=False)
    df.loc[:, cols] = df_sub.where(np.abs((df_sub - df_sub.median()) / iqr) < 2.22, np.nan)    
    '''
    for col in cols:
        if col.find('LM')!=-1:
            continue
        df.loc[:,col] = df_sub.where(df_sub[col]!=0.0,np.nan)
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
    plt.ylim(-0.05,180.05)
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
    
def filter_and_plot_singleFile(filename,thresh,which):
    plt.rc("figure",figsize=(10,6))
    plt.rc("axes",prop_cycle= plt.cycler("color", plt.cm.tab20.colors))
    
    df = read_data(filename,which)
    # remove outlier samples from devices
    reject_outliers(df)

    # remove noisy devices
    filter_noisy(df,thresh,True)
    
    # plot raw BPM phase data
    x = (len(df.keys())-1)
    for i in range(int(x/12.)+1):
        if which=='phase':
            plot_phases_norm(df,i*12,min(x,(i+1)*12))
        else:
            plot_raw(df,i*12,min(x,(i+1)*12),which)
    
    #plot averages
    plot_one_avg(df,which)


def filter_and_plot_multiFile(files,REF,thresh,which):
    plt.rc("figure",figsize=(20,12))
    overlap = {name for name in mcolors.CSS4_COLORS
               if f'xkcd:{name}' in mcolors.XKCD_COLORS}

    overlap.difference_update(['aqua','ivory','white','lime','chocolate','gold'])
    colors = [mcolors.XKCD_COLORS[f'xkcd:{color_name}'].upper() for color_name in sorted(overlap)]    
    labels = get_labels(files)
    lbls = [l.split(' ')[1] for l in labels]

    listdf = [None]*len(files)
    
    for i in range(len(files)):
        listdf[i] = read_data(files[i],which)

    refdf = read_data(REF,which)
    colnames = [col for col in list(refdf.keys()) if col.find('Time')==-1] # reference

    numcol = int(len(listdf)/2) if len(listdf)%2==0 else int(len(listdf)/2)+1
    fig, axs = plt.subplots(numcol,2,sharex=True, sharey=True)

    print(len(listdf), len(listdf)/2)

    list_delta_dict = []

    reject_outliers(refdf)
    filter_noisy(refdf,thresh,False)
    
    for k,df in enumerate(listdf):
        # remove outlier samples
        reject_outliers(df)
        # remove noisy
        filter_noisy(df,thresh,False)

        # initialize delta_dict
        delta_dict = {}

        delta = []
        idx = []
        for l,col in enumerate(colnames):
            date = files[k].split('/')[-1]
            date = date.split('_devicescan.csv.zip')[0]

            delta_dict.update({col : 0.0})
            
            if col in list(df.keys()) and col in list(refdf.keys()):
                if len(df[col])>0 and len(refdf[col])>0:
                    if which=='blm':
                        delta.append(np.median(df[col]))
                        #delta.append(np.mean(df[col]))
                    else:
                        delta.append(np.median(df[col]) - np.median(refdf[col]))
                        #delta.append(np.mean(df[col]) - np.mean(refdf[col]))
                        #if (which=='phase' and col.find('2OF')!=-1) or (which=='bph' and col.find('2OT')!=-1) or (which=='bpv' and col.find('2OT')!=-1):
                        #    print('when: ',date,'device: ',col, 'ref: ',np.mean(refdf[col]), 'now: ',np.mean(df[col]))

                    idx.append(l)

        [delta_dict.update({colnames[idxs] : delta[i]}) for i,idxs in enumerate(idx)]
        list_delta_dict.append(delta_dict)

        axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].fill_between(idx,delta,facecolor=colors[k],label='%s'%lbls[k])
        axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].legend(loc='upper right', fancybox=True, fontsize='xx-large')
        axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].xaxis.set_tick_params(direction='in', which='major')
        axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].grid(True)
            
        if which=='phase':
            #axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].fill_between(idx,delta,facecolor='k',alpha=0.7,label='%s'%lbls[k])
            axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].set_ylim(-1.5,1.5)
            fig.supylabel(' $\Delta$ Phase (deg)',fontsize=24)
            fig.suptitle('BPM phases',fontsize=24)
        elif which=='bpv':    
            axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].set_ylim(-0.4,0.4)
            fig.supylabel('$\Delta$ Y (mm)',fontsize=24)    
            fig.suptitle("BPM Vertical positions",fontsize=24)
        elif which=='bph':
            axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].set_ylim(-0.5,0.5)
            fig.supylabel('$\Delta$ X (mm)',fontsize=24)    
            fig.suptitle("BPM Horizontal positions",fontsize=24)
        elif which=='blm':
            axs[int(k%(len(listdf)/2))][int(k/(len(listdf)/2))].set_ylim(0,3)
            fig.supylabel('Loss (cnt)',fontsize=24)
            fig.suptitle('BLMs',fontsize=24)

    axs[numcol-1][0].set_xticks(np.arange(len(colnames)),colnames, rotation = 'vertical',fontsize=14)
    axs[numcol-1][1].set_xticks(np.arange(len(colnames)),colnames, rotation = 'vertical',fontsize=14)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(left=0.06)
    plt.subplots_adjust(right=0.98)
    #plt.show()

    plt.savefig('%s_%s_%s.png'%(labels[-1].split(' ')[0].replace('/','-'),labels[-1].split(' ')[1].replace(':','-'),which))
    deltadf = pd.DataFrame.from_dict(list_delta_dict)
    deltadf.to_csv(r'%s_%s_%s.csv'%(labels[-1].split(' ')[0].replace('/','-'),labels[-1].split(' ')[1].replace(':','-'),which),index=False,header=True)


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

def get_one_file(path,date):
    files =[]
    datestr= date.strftime('%Y_%m_%d-%H')

    for subdir, dirs, fls in os.walk(path):
        for file in fls:
            if datestr in file:
                files.append(os.path.join(subdir,file))
    files.sort()
    return files[0]


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
    #parser.add_argument ('--o',  dest='imgdir', default='',
    #                     help="Image save directory")
    parser.add_argument ('--d', dest='date', default='',
                         help="End date/time to plot. Format %%Y_%%m_%%d-%%H_%%M_%%S")
    parser.add_argument ('--r', dest='ref', default='',
                         help="Reference date/time. Format %%Y_%%m_%%d-%%H_%%M_%%S")
    
    
    options = parser.parse_args()
    datadir = options.datadir
    #imgdir  = options.imgdir
    datestr = options.date
    refstr  = options.ref

    if datestr =='':
        date = datetime.today()
    else:
        try:
            date = datetime.strptime(datestr,'%Y_%m_%d-%H_%M_%S')
        except:
            print('Invalid date format. Using default.')
            date = datetime.today()

    if refstr =='':
        ref = datetime.today().replace(hour=0, minute=0, second=0)
    else:
        try:
            ref = datetime.strptime(refstr,'%Y_%m_%d-%H_%M_%S')
        except:
            print('Invalid date format. Using default.')
            ref = datetime.today().replace(hour=0, minute=0, second=0)

    return datadir,date,ref

def main():
    
    path,date,ref = parse_args()
    REF = get_one_file(path,ref)
    files = get_files(path,date)

    filter_and_plot_multiFile(files,REF,10,'phase')
    filter_and_plot_multiFile(files,REF,2,'blm')
    filter_and_plot_multiFile(files,REF,1.5,'bph')
    filter_and_plot_multiFile(files,REF,1.5,'bpv')

    #filter_and_plot_singleFile(files[4],1,3,'bph')
    

if __name__ == "__main__":
    main()
