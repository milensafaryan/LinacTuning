import io
import os,sys,platform
import numpy as np
import matplotlib.pyplot as plt
import functools

def get_colnames(filename,which):
    file = io.open(r'%s'%filename, 'r',encoding="utf-8")
    colnames = file.readlines()[31]
    if which=='phase':
        colnames = [col for col in colnames.strip('\n').split(',') if col.find('F')!=-1 and col.find('PAH')==-1 and col.find('SS')==-1 and col.find('DRQF')==-1 ]
    elif which=='blm':
        colnames = [col for col in colnames.strip('\n').split(',') if col.find('LM')!=-1]
    elif which=='bpv':
        colnames = [col for col in colnames.strip('\n').split(',') if col.find('BPV')!=-1]
    elif which=='bph':
        colnames = [col for col in colnames.strip('\n').split(',') if col.find('BPH')!=-1]
    elif which=='all':
        print('Warning: grabbing all columns')
    else:
        colnames = []
        print('Invalid argument')
    file.close()
    return colnames

def read_data(filename,which):
    file = io.open(r'%s'%filename, 'r',encoding="utf-8")
    lines = file.readlines()[31:]
    idx= []
    if which=='phase': 
        idx = [i for i,col in enumerate(lines[0].strip('\n').split(',')) if col.find('F')!=-1 and col.find('PAH')==-1 and col.find('SS')==-1 and col.find('DRQF')==-1]
    elif which=='blm':
        idx = [i for i,col in enumerate(lines[0].strip('\n').split(',')) if col.find('LM')!=-1]
    elif which=='bpv':
        idx = [i for i,col in enumerate(lines[0].strip('\n').split(',')) if col.find('BPV')!=-1]
    elif which=='bph':
        idx = [i for i,col in enumerate(lines[0].strip('\n').split(',')) if col.find('BPH')!=-1]
    elif which=='all':
        idx = [i for i in range(len(lines[0].strip('\n').split(',')))]
        print('Warning: grabbing all columns')
    else:
        colnames = []
        print('Invalid argument')

        
    x = [[] for i in idx] #diagnostic
    t = [[] for i in idx] #time

    for line in lines[1:]:
        cols = [j for j in line.strip('\n').split(',')]
        ts = cols[1]
        cols = [cols[k] for k in idx]

        for i,col in enumerate(cols):
            if col!='':
                t[i].append(float(ts))
                x[i].append(float(col))

    file.close()
    return t,x


def filter_noisy(x,threshold,verbose):
    # Note: x[i] is a 2D array of shape (Ndet,Nsample)
    idx = [i for i in range(len(x)) if np.var(x[i])<threshold]
    if verbose:
        [ print(np.var(x[i])) for i in range(len(x)) ]
        print('idx: ',idx)
    return idx

def reject_outliers(t,x,imin,imax,m=3):
    for i in range(imin,imax):
        idx = [j for j,xj in enumerate(x[i]) if abs(xj - np.median(x[i])) < m * np.std(x[i])]
        t[i] = [t[i][j] for j in idx]
        x[i] = [x[i][j] for j in idx]

    return t,x

def calc_delta(xref,x,idx):
    xref_m = [np.mean(xref[i]) for i in idx]
    x_m = [np.mean(x[i]) for i in idx]

    dx = np.subtract(x_m,xref_m)
    return dx
    
def plot_raw(t,x,colnames,start,stop,which):
    plt.grid(color='k', linestyle='-', linewidth=1.)
    plt.xlabel('Timestamp')

    if which=='phase':
        plt.ylim(-185,185)
        plt.ylabel('Phase (deg)')
        plt.title('BPM phases')
    elif which=='bpv':    
        plt.ylim(-5.,5)
        plt.ylabel('Y (mm)')    
        plt.title("BPM Vertical positions")
    elif which=='bph':
        plt.ylim(-5.,5)
        plt.ylabel('X (mm)')    
        plt.title("BPM Horizontal positions")
    elif which=='blm':
        plt.ylim(-1,5)
        plt.ylabel('Loss (cnt)')
        plt.title('BLMs')
        
    [ plt.plot(t[i],x[i],label='%s'%colnames[i]) for i in range(int(start),int(stop))]
    plt.legend()
    plt.show()

def plot_phases_norm(t,x,colnames,start,stop):
    plt.grid(color='k', linestyle='-', linewidth=1.)
    plt.ylim(-0.05,2.55)
    plt.xlabel('Timestamp')
    plt.ylabel('Phase (normalized)')
    plt.title("BPM Phases")

    for i in range(int(start),int(stop)):
        xnorm = np.asarray(x[i])
        try:
            #xnorm = (xnorm - xnorm.min(0)) / xnorm.ptp(0)
            xnorm = (xnorm - xnorm.min(0))
        except ValueError:
            pass
        
        plt.plot(t[i][:],xnorm[:],label='%s'%colnames[i])

    plt.legend()
    plt.show()
                         
def plot_one_dx(idxm,dxm,idxp,dxp,colnames):
    plt.plot(idxp,dxp,marker = '', c = 'g',label='Plus 1 deg')
    plt.plot(idxm,dxm,marker = '', c = 'b',label='Minus 1 deg')
    ticks = [i for i in range(len(colnames))]
    plt.xticks(ticks,colnames, rotation = 'vertical')
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

def osc():    
    names = ['reference','rfq_minus5deg','rfq_plus5deg','rfb_minus1deg','rfb_plus1deg']
    for i in range(1,6):
        names.append('tank%d_minus1deg'%i)
        names.append('tank%d_plus1deg'%i)

    path = r'/Users/rshara01-local/Desktop/LINAC_STUDY/BEAM_STUDY_21FEB2022/'
    files = [ '%s%s.csv'%(path,names[i]) for i in range(len(names))] 

    colnames = get_colnames(files[0],'phase')
    tref,xref = read_data(files[0],'phase')
    tm,xm,idxm,dxm = [None]*7,[None]*7,[None]*7,[None]*7
    tp,xp,idxp,dxp = [None]*7,[None]*7,[None]*7,[None]*7
    
    for i in range(1,8):
        tm[i-1],xm[i-1] = read_data(files[i*2-1],'phase')
        tp[i-1],xp[i-1] = read_data(files[i*2],'phase')
        
        idxm[i-1] = filter_noisy(xm[i-1],1,False)
        idxp[i-1] = filter_noisy(xp[i-1],1,False)

        dxm[i-1] = calc_delta(xref,xm[i-1],idxm[i-1])
        dxp[i-1] = calc_delta(xref,xp[i-1],idxp[i-1])

    plot_all_dx(idxm,dxm,idxp,dxp,colnames)

def plot_one_avg(x,idx,colnames,which):
    try:
        x_m = [np.mean(xx) for xx in x if len(xx)>0]
        idx = [idx[j] for j,xx in enumerate(x) if len(xx)>0 ]
    except ValueError:
        pass

    plt.grid(color='k', linestyle='-', linewidth=1.)
    plt.plot(x_m,marker = '', c = 'g',label='')

    if which=='phase':
        plt.ylim(-185,185)
        plt.ylabel('Phase (deg)')
        plt.title('BPM phases')
    elif which=='bpv':    
        plt.ylim(-6.,6)
        plt.ylabel('Y (mm)')    
        plt.title("BPM Vertical positions")
    elif which=='bph':
        plt.ylim(-5.,5)
        plt.ylabel('X (mm)')    
        plt.title("BPM Horizontal positions")
    elif which=='blm':
        plt.ylim(-1,5)
        plt.ylabel('Loss (cnt)')
        plt.title('BLMs')

    colnames = [colnames[i] for i in idx]
    plt.xticks(idx,colnames, rotation = 'vertical')
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(top=0.94)
    plt.subplots_adjust(left=0.11)
    plt.subplots_adjust(right=0.95)

    plt.show()
    
def filter_and_plot_singleFile(filename,thresh,m,which):
    colnames = get_colnames(filename,which)
    t,x = read_data(filename,which)
    # remove noisy
    idx = filter_noisy(x,thresh,False)
    t = [t[i] for i in idx]
    x = [x[i] for i in idx]
    colnames = [colnames[i] for i in idx]
    # remove outlier samples
    t,x = reject_outliers(t,x,0,len(x),m)
    
    # plot raw BPM phase data
    for i in range(int(len(x)/9.)+1):
        if which=='phase':
            plot_phases_norm(t,x,colnames,i*9,min(len(x),(i+1)*9))
        else:
            plot_raw(t,x,colnames,i*9,min(len(x),(i+1)*9),which)
    
    #plot averages
    plot_one_avg(x,idx,colnames,which)


def filter_and_plot_multiFile(files,thresh,m,which,REF):
    plt.rcParams["figure.figsize"] = (10,6)
    
    colnames = get_colnames(files[0], which)
    labels = get_labels(files)

    kk = int([i for i,s in enumerate(labels) if REF in s][0])
            
    listt = [None]*len(files)
    listx = [None]*len(files)
    x_m   = [None]*len(files)
    idx_m = [None]*len(files)
    
    for i in range(len(files)):
        listt[i],listx[i] = read_data(files[i],which)

    for k,(t,x) in enumerate(zip(listt,listx)):
        # remove noisy
        idx = filter_noisy(x,thresh,False)
        t = [t[i] for i in idx]
        x = [x[i] for i in idx]
        # remove outlier samples
        t,x = reject_outliers(t,x,0,len(x),m)
    
        #get averages
        x_m[k] = [np.mean(xx) for xx in x if len(xx)>0]
        idx_m[k] = [idx[j] for j,xx in enumerate(x) if len(xx)>0 ]
    
    for k,x in enumerate(x_m):
        if k!=kk:
            common = [idx for idx in idx_m[k] if idx in idx_m[kk]]
            x = [x[i] for i,idx in enumerate(idx_m[k]) if idx in common]
            xref = [x_m[kk][i] for i,idx in enumerate(idx_m[kk]) if idx in common]
            plt.plot(common,np.subtract(x,xref),marker = '.',label='%s'%labels[k])

                
    plt.grid(color='k', linestyle='-', linewidth=1.)

    if which=='phase':
        plt.ylim(-5,5)
        plt.ylabel('Phase (deg)')
        plt.title('BPM phases')
    elif which=='bpv':    
        plt.ylim(-1.,1)
        plt.ylabel('Y (mm)')    
        plt.title("BPM Vertical positions")
    elif which=='bph':
        plt.ylim(-1.,1)
        plt.ylabel('X (mm)')    
        plt.title("BPM Horizontal positions")
    elif which=='blm':
        plt.ylim(-1,1)
        plt.ylabel('Loss (cnt)')
        plt.title('BLMs')
    
    ticks = [i for i in range(len(colnames))]
    plt.xticks(ticks,colnames, rotation = 'vertical')
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(top=0.94)
    plt.subplots_adjust(left=0.11)
    plt.subplots_adjust(right=0.95)
    plt.legend(loc='best',ncol = 4, fancybox=True, fontsize='small')
    plt.show()


def get_files(path):
    files =[]
    for subdir, dirs, fls in os.walk(path):
        for file in fls:
            if r'BEAM STUDY' in subdir and r'correlatedPulseData' in file:
                files.append(os.path.join(subdir, file))
    files.sort()
    return files

def get_labels(files):
    labels = []
    for f in files:
        if platform.system()=='Windows':
            labels.append(''.join([f.split('BEAM STUDY ')[-1].split('correlatedPulseData')[0].strip('\\'),
                                   f.split('BEAM STUDY ')[-1].split('correlatedPulseData')[-1].strip('.csv')]))
        else:
            labels.append(''.join([f.split('BEAM STUDY ')[-1].split('correlatedPulseData')[0].strip('/'),
                                   f.split('BEAM STUDY ')[-1].split('correlatedPulseData')[-1].strip('.csv')]))

    return labels

def check_col_order(files,which):
    lists =[get_colnames(f,which) for f in files]

    for i in range(len(lists)-1):
        if functools.reduce(lambda x, y: x and y, map(lambda a, b: a == b, lists[i], lists[i+1]), True):
            continue
        else:
            print("Column names not in the same order! Check input data.")

def main():

    path = r'/Users/rshara01-local/Desktop/LINAC_STUDY/'
    files = get_files(path)

    #filter_and_plot_multiFile(files,1,2.5,'phase','07MAR2022')
    #filter_and_plot_multiFile(files,1,2.5,'blm','07MAR2022')
    filter_and_plot_multiFile(files,1,2.5,'bph','07MAR2022')
    #filter_and_plot_multiFile(files,1,2.5,'bpv','07MAR2022')

    

if __name__ == "__main__":
    main()
