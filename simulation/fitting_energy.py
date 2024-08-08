import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.mixture

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors


plt.rcParams.update({'font.size': 16,
                     'mathtext.fontset': 'cm', 'savefig.format' : 'pdf'})
# Define constants
proton_mass=938272088.16
electron_mass=510998.95
pmass= proton_mass+2*electron_mass
light_v=299792458
def interpolate_E(energy,Ncells):
    interpolated = []

    if Ncells>1:
        for i in range(len(energy) - 1):
            increment = (energy[i+1] - energy[i]) / Ncells
            for j in range(Ncells):
                interpolated.append(energy[i] + j * increment)
        interpolated.append(energy[-1])
    else:
        interpolated = energy

    return interpolated
def calc_dE(energy):
    dE = []
    if len(energy) >=2:
        for i in range(len(energy) - 1):
            dE.append(energy[i+1] - energy[i])
    dE.append(0)

    return dE
def calc_grad(energy,synch_phase):
    grad = np.divide(energy,np.cos(synch_phase/180*np.pi))

    return grad

#calculates voltage
def calc_beta(gamma):
    beta = np.sqrt(1 - np.divide(1,np.array(gamma**2)))

    return beta
def calc_gamma(energy,mass):
    gamma = np.divide(np.add(energy,mass),mass)

    return gamma
def calc_L(beta,rf_freq,cavtype):
    light_v=299792458

    L=[]
    if cavtype=='SCL':
        L = np.multiply(beta,light_v/rf_freq/2)
        #Why we divide by 2
    elif cavtype=='DTL':
        L = np.multiply(beta,light_v/rf_freq)
    else:
        L = np.multiply(beta,light_v/rf_freq) # need to edit for fixed-lenght transition section

    #print(beta[:10],L[:10])
    return L
def make_section(cavtype,energy,synch_phase,mass,rf_freq,Ncells,Nmodules):

    E = interpolate_E(energy,Ncells)
    dE = calc_dE(E)
    V = calc_grad(dE,synch_phase)
    G = calc_gamma(E,mass)
    B = calc_beta(G)
    L = calc_L(B,rf_freq,cavtype)

    df = pd.DataFrame({'Energy_cell' : E,
                      'dE' : dE,
                      'V' : V,
                      'Gamma' : G,
                      'Beta' : B,
                      'L' : L,
                      'Phi' : synch_phase,
                      'RF' : rf_freq,
                      'Type' : cavtype})
    if cavtype=='SCL':
        # Add drifts
        for i in range(1,Nmodules):
            n=i*Ncells-1
            df.loc[n+0.5]=df.loc[n+1]
            df.loc[n+0.5,'V']=0
            df.loc[n+0.5,'dE']=0
            df.loc[n+0.5,'L']=float(df.loc[n+0.5,'L']*4)
            #does there drift tube between everycell, if not our calculated L of distances, why its based on the fact that L = VT(I assume vwlocity does not change here)
        df = df.sort_index().reset_index(drop=True)
    return df
def fetch_data(file,datacols,cuts,setdevs):
    dataset = pd.read_csv(file)
    dataset.columns = dataset.columns.str.replace("[()]", "_",regex=True)

    cols = list(dataset.filter(regex='|'.join(datacols)))

    # for set points, keep _S_ and drop _R_ if available
    cols = [col for col in cols for word in setdevs if col.find(word)==-1]

    subset = dataset.loc[:,cols]
    subset.columns = subset.columns.str.replace("_R_|_S_", "",regex=True)
    subset.drop(list(subset.filter(regex=r'\.1|Time|step|iter')),axis=1, inplace=True)

    # apply data quality cuts
    subset.query(cuts,inplace=True)

    # augment jumps in phase data
    #subset['B:BQ3F'] = subset['B:BQF3'].apply(lambda x : x if x > 0 else x +360)
    subset.dropna()

    print(subset.head())

    return subset


# SCL Energy and energy gain per cell from design
energy = np.multiply([357.1,368.1,379.1,390.2,401.5],1e6)
df = make_section('SCL',energy,-32,pmass,805e6,16,4)
from scipy.optimize import curve_fit

def fitting_curve_dispersion(file):
  dataset_test = pd.read_csv(file)
  dataset1 = fetch_data(file,['L7PADJ','HPQ3','HPQ4','HPQ5','VPQ5','D64BF','D74BF'],'`L:L7PADJ`>54 & `L:L7PADJ`<96',['L7PADJ_R'])
  ref = (dataset1.loc[(np.abs(dataset1['L:L7PADJ']-82.)<0.205) & (dataset1['L:L7PADJ']>=82.)]).mean()
  dataset1 = dataset1 - ref
  #D = [-0.4997,-2.224,-1.406,0.98], old dispersion factors
  E_kin=401.5e6
  E_t=E_kin+938e6
  beta=0.7136

  D_new = [- 0.37, -1.6, -1.19, 0.66]

  dataset1['B:HPQ3_E']=dataset1['B:HPQ3']/1000/D_new[0]*beta*beta*E_t+E_kin
  dataset1['B:HPQ4_E']=dataset1['B:HPQ4']/1000/D_new[1]*beta*beta*E_t+E_kin
  dataset1['B:HPQ5_E']=dataset1['B:HPQ5']/1000/D_new[2]*beta*beta*E_t+E_kin
  dataset1['B:VPQ5_E']=dataset1['B:VPQ5']/1000/D_new[3]*beta*beta*E_t+E_kin

  dfavg= dataset1.groupby(np.arange(len(dataset1))//10).mean()


  dfavg = dfavg.iloc[2:-2]
  #extracting bad edge points
  #plt.scatter(dfavg['L:L7PADJ'],dfavg['B:HPQ3_E'], label='HPQ3',marker='.',color='k')
  #plt.scatter(dfavg['L:L7PADJ'],dfavg['B:HPQ4_E'], label='HPQ4',marker='.',color='b')
  #plt.scatter(dfavg['L:L7PADJ'],dfavg['B:HPQ5_E'], label='HPQ5',marker='.',color='r')

  #plt.xlabel('$\Delta\phi_{RF}$ (deg)',fontsize='x-large')
  #plt.ylabel('$E_{out}$ (eV)',fontsize='x-large')
  #plt.legend(loc='lower right')
  #plt.show()
  
  mean_output_energy= (dfavg['B:HPQ3_E'] + dfavg['B:HPQ4_E'] + dfavg['B:HPQ5_E'])/3

  #plt.scatter(dfavg['L:L7PADJ'], mean_output_energy, color = "black")
  #plt.show()

  #plt.scatter(dfavg['L:L7PADJ'] - min(dfavg['L:L7PADJ']) , mean_output_energy, color = "black")
  #shifted
  x = dfavg['L:L7PADJ'] - min(dfavg['L:L7PADJ'])

  def poly(x, a, b, c):
    z = a * (x - b)**2 + c 
    return z
  initial_a = (mean_output_energy[5] - max(mean_output_energy))/ (x[5] + max(x) - min(x) )**2
  initial_parameters = [initial_a, max(x) - min(x), max(mean_output_energy) * 10**8]
  popt,pcov = curve_fit(poly, x, mean_output_energy, p0=initial_parameters)


  a, b, c = popt
  fitting_energy = poly(x,a,b,c)
  #plt.scatter(dfavg['L:L7PADJ'] - min(dfavg['L:L7PADJ']), mean_output_energy, color = "black")
  #plt.scatter(x, fitting_energy )

  #plt.show()

  given_phase_change = dfavg['L:L7PADJ']
  given_energies = fitting_energy
  #plt.plot(given_phase_change, given_energies)
  #plt.scatter(dfavg['L:L7PADJ'],dfavg['B:HPQ3_E'], label='HPQ3',marker='.',color='k')
  #plt.scatter(dfavg['L:L7PADJ'],dfavg['B:HPQ4_E'], label='HPQ4',marker='.',color='b')
  #plt.scatter(dfavg['L:L7PADJ'],dfavg['B:HPQ5_E'], label='HPQ5',marker='.',color='r')
  #plt.show()
  return ([a, b, c], given_phase_change, given_energies )

file1 = r'C:\Users\safaryan\work\devicescan_TrainingData_Feb202023.csv'
file2 = r'C:\Users\safaryan\work\devicescan2.csv'

a1 = fitting_curve_dispersion(file2)
a2 = fitting_curve_dispersion(file1)



plt.plot(a1[1], a1[2], label='New data', color='blue')
plt.plot(a2[1], a2[2], label='Old data', color='red')
plt.show()
