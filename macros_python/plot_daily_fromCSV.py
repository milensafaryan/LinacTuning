import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#plt.rc("figure",figsize=(20,12))
plt.rc("figure",figsize=(12,8))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["savefig.format"] = 'pdf'


date='2022-05-22_00-00'
which = ['phase','bph','bpv']

listdf = []

for w in which:
    listdf.append(pd.read_csv('%s_%s.csv'%(date,w)))


for i,w in enumerate(which):
    cols = list(listdf[i].columns)
    idx = listdf[i].columns.get_indexer(cols)
    # t1out t2out t3out t4out t5out
    # m11 m12 m13 m14
    # m21 m22 m23 m24
    # m31 m32 m33 m34
    # m41 m42 m43 m44
    # m51 m52 m53 m54
    # m61 m62 m63 m64
    # m71 m72 m73 m74
    dist = [7.6, 26.5, 43.5, 60.8, 77.0,
            80.8, 82.5, 84.4, 86.3,
            88.2, 90.1, 92.1, 94.2,
            96.3, 98.4, 100.6, 102.8,
            105.0, 107.3, 109.6, 111.9,
            114.7, 116.7, 119.1, 121.6,
            124.0, 126.5, 129.6, 131.6,
            134.2, 136.8, 139.4, 141.6]
    dist_data = [26.5,27.0, 43.5, 44.0,60.8, 77.0,78.3,79.6,
            80.8, 82.5, 84.4,
            88.2, 90.1, 92.1,
            96.3, 98.4, 100.6, 102.8,
            105.0, 107.3, 109.6, 111.9,
            114.7, 116.7, 119.1, 121.6,
            124.0, 126.5, 129.6, 131.6,
                 134.2, 136.8, 139.4, 141.6]
    #[print(x) for x in zip(cols,dist_data)]
    numcol = int(len(listdf[i])/2) if len(listdf[i])%2==0 else int(len(listdf[i])/2)+1
    numcol = 10
    fig, axs = plt.subplots(numcol,1,sharex=True, sharey=True)

    #print(cols)
    
    for k in range(numcol):
        
        if w=='phase':
            '''
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].fill_between(idx,listdf[i].iloc[k,:],facecolor='k',alpha=0.8,label='%d:00'%k)
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].set_ylim(-1.5,1.5)
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].legend(loc='upper right', fancybox=True, fontsize='medium')
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].xaxis.set_tick_params(direction='in', which='major')
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].grid(True)
            '''
            #axs[k].fill_between(idx,listdf[i].iloc[k,:],facecolor='k',alpha=0.9,label='%d:00'%k)
            axs[k].scatter(dist_data,listdf[i].iloc[k,:],color='k',linestyle='-',label='%d:00'%k)
            axs[k].set_xlim(0,150)
            axs[k].set_ylim(-1.5,1.5)
            #axs[k].legend(loc='upper right', fancybox=True, fontsize='xx-large')
            axs[k].xaxis.set_tick_params(direction='in', which='major')
            axs[k].tick_params(axis='both',labelsize=17)
            #axs[k].tick_params(axis='y',labelsize=15)
            axs[k].grid(True)
            axs[k].set_xlabel('Distance (m)', fontsize='18')
            axs[k].text(0.955, 0.75, '%d:00'%(k+1), transform=axs[k].transAxes, color='k', alpha=0.99,fontsize='x-large',verticalalignment='top')
            axs[0].text(0.4, 0.6, 'REFERENCE', transform=axs[0].transAxes, color='k',fontsize='xx-large',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.99))
            fig.supylabel(' $\Delta \phi$ (deg)',fontsize=20)
            fig.suptitle('BPM phases',fontsize=20)
            
        elif w=='bph':
            #continue
            '''
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].fill_between(idx,listdf[i].iloc[k,:],facecolor='r',label='%2.2f'%k)
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].set_ylim(-0.5,0.5)
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].legend(loc='upper right', fancybox=True, fontsize='xx-large')
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].xaxis.set_tick_params(direction='in', which='major')
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].grid(True)

            fig.supylabel(' $\Delta$ X (mm)',fontsize=20)
            fig.suptitle('BPM horizontal positions',fontsize=20)
            '''
        elif w=='bpv':
            '''
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].fill_between(idx,listdf[i].iloc[k,:],facecolor='b',alpha=0.65,label='%d:00'%k)
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].fill_between(idx,listdf[i-1].iloc[k,:],facecolor='r',alpha=0.65)
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].set_ylim(-0.4,0.4)
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].legend(loc='upper right', fancybox=True, fontsize='medium')
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].xaxis.set_tick_params(direction='in', which='major')
            axs[int(k%(len(listdf[i])/2))][int(k/(len(listdf[i])/2))].grid(True)
            '''
            # axs[k].scatter(dist_data,listdf[i].iloc[k,:],color='b', linestyle='-',label='%d:00'%k)
            axs[k].bar(dist_data,listdf[i].iloc[k,:], width=1.1,color='b', alpha=0.75,label='%d:00'%k)
            #axs[k].scatter(dist_data,listdf[i-1].iloc[k,:],color='r',label='%d:00'%k)
            axs[k].bar(dist_data,listdf[i-1].iloc[k,:], width=1.1,color='r', alpha=0.75,label='%d:00'%k)
            #axs[k].fill_between(dist_data,listdf[i-1].iloc[k,:],facecolor='r',alpha=0.75)
            axs[k].set_ylim(-0.25,0.25)
            #axs[k].legend(loc='upper right', fancybox=True, fontsize='xx-large')
            axs[k].xaxis.set_tick_params(direction='in', which='major')
            axs[k].tick_params(axis='x',labelsize=15)
            axs[k].tick_params(axis='y',labelsize=15)
            axs[k].grid(True)
            axs[k].text(0.955, 0.75, '%d:00'%(k+1), transform=axs[k].transAxes, color='k', alpha=0.99,fontsize='xx-large',verticalalignment='top')
            
            axs[0].text(0.4, 0.6, 'REFERENCE', transform=axs[0].transAxes, color='k', alpha=0.99,fontsize='xx-large',verticalalignment='top')

            str1 = r'$\Delta$X'
            str2 = r'$\Delta$Y'
            axs[0].text(0.05, 0.9, str1, transform=axs[0].transAxes, color='r', alpha=0.65,fontsize='xx-large',verticalalignment='top')
            axs[0].text(0.05, 0.5, str2, transform=axs[0].transAxes, color='b', alpha=0.65,fontsize='xx-large',verticalalignment='top')

            #axs[0][0].text(0.05, 0.9, str1, transform=axs[0][0].transAxes, color='r', alpha=0.65,fontsize='x-large',verticalalignment='top')
            #axs[0][0].text(0.05, 0.5, str2, transform=axs[0][0].transAxes, color='b', alpha=0.65,fontsize='x-large',verticalalignment='top')
            fig.supylabel(' $\Delta$X and $\Delta$Y (mm)',fontsize=20)
            fig.supxlabel('Distance along linac (m)',fontsize=20)
            #fig.suptitle('BPM positions',fontsize=20)
            
            
    #axs[numcol-1][0].set_xticks(idx,cols, rotation = 'vertical',fontsize=12)
    #axs[numcol-1][1].set_xticks(idx,cols, rotation = 'vertical',fontsize=12)

    #axs[numcol-1].set_xticks(idx,cols, rotation = 'vertical',fontsize='xx-large')
    #axs[numcol-1].set_xticks(idx,dist_data,rotation='vertical',fontsize='xx-large')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(bottom=0.10)
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(left=0.09)
    plt.subplots_adjust(right=0.98)
    plt.show()

