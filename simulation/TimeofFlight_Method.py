
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file=r'//devicescan2.csv'
dataset_test = pd.read_csv(file)
dataset_test = pd.DataFrame(dataset_test)
desired_rows = ['L:L7PADJ(R)', 'B:BQ2F(R)', 'B:BQ3F(R)', 'B:BQ4F(R)', 'B:BQ5F(R)']
filtered_dataset = dataset_test[desired_rows]
filtered_dataset = filtered_dataset.iloc[4: , :]
print(filtered_dataset)
bpm_phase = 4.02e8
speed_of_light = 3e8
spane_of_phase = 360
E0 = 939
distances = [4.6913, 8.8677, 13.2251]


total_turn = [9,17, 25]
filtered_dataset['B:BQ2F(R)'] = filtered_dataset['B:BQ2F(R)'] +141.7
filtered_dataset['B:BQ3F(R)'] = filtered_dataset['B:BQ3F(R)'] - 149.5
filtered_dataset['B:BQ4F(R)'] = filtered_dataset['B:BQ4F(R)'] - 97.24
filtered_dataset['B:BQ5F(R)'] = filtered_dataset['B:BQ5F(R)'] - 52.3
for k,i in enumerate(['B:BQ3F(R)','B:BQ4F(R)', 'B:BQ5F(R)']):
  filtered_dataset[f'Phase Change + {k}'] = (filtered_dataset[i] - filtered_dataset['B:BQ2F(R)'])/spane_of_phase
  filtered_dataset[f'oscillation number + {k}'] = filtered_dataset[f'Phase Change + {k}'] + total_turn[k]
  filtered_dataset[f'total_phase + {k}'] = filtered_dataset[f'oscillation number + {k}'] * spane_of_phase
  filtered_dataset[f'total_time + {k}'] = filtered_dataset[f'total_phase + {k}'] / spane_of_phase/bpm_phase
  filtered_dataset[f'betta + {k}'] = distances[k]/filtered_dataset[f'total_time + {k}']/speed_of_light
  filtered_dataset[f'gamma + {k}'] = 1/np.sqrt(1-filtered_dataset[f'betta + {k}']*filtered_dataset[f'betta + {k}'])
  filtered_dataset[f'{i}+ _E'] = filtered_dataset[f'gamma + {k}'] * E0 -E0
pd.to_numeric(filtered_dataset['L:L7PADJ(R)'], errors='coerce')
plt.scatter(pd.to_numeric(filtered_dataset['L:L7PADJ(R)'], errors='coerce') - pd.to_numeric(filtered_dataset['L:L7PADJ(R)'], errors='coerce').mean() ,filtered_dataset['B:BQ3F(R)+ _E'] , color = "red")
plt.scatter(pd.to_numeric(filtered_dataset['L:L7PADJ(R)'], errors='coerce') - pd.to_numeric(filtered_dataset['L:L7PADJ(R)'], errors='coerce').mean(),filtered_dataset['B:BQ4F(R)+ _E'],color = "blue")
plt.scatter(pd.to_numeric(filtered_dataset['L:L7PADJ(R)'], errors='coerce') - pd.to_numeric(filtered_dataset['L:L7PADJ(R)'], errors='coerce').mean() ,filtered_dataset['B:BQ5F(R)+ _E'] ,color = "orange")

#dfavg2= filtered_dataset.groupby(np.arange(len(filtered_dataset))//10).mean()


#plt.scatter(dfavg2['L:L7PADJ(R)'] - np.average(dfavg2['L:L7PADJ(R)']),dfavg2['B:BQ3F(R)+ _E'] )
#plt.scatter(dfavg2['L:L7PADJ(R)']  - np.average(dfavg2['L:L7PADJ(R)']),dfavg2['B:BQ4F(R)+ _E'])
#plt.scatter(dfavg2['L:L7PADJ(R)']  - np.average(dfavg2['L:L7PADJ(R)']),dfavg2['B:BQ5F(R)+ _E'] )
#plt.scatter(filtered_dataset['L:L7PADJ(R)'] - np.average(filtered_dataset['L:L7PADJ(R)']),filtered_dataset['B:BQ3F(R)+ _E'] , color = "red")
#plt.scatter(filtered_dataset['L:L7PADJ(R)']  - np.average(filtered_dataset['L:L7PADJ(R)']),filtered_dataset['B:BQ4F(R)+ _E'],color = "blue")
#plt.scatter(filtered_dataset['L:L7PADJ(R)']  - np.average(filtered_dataset['L:L7PADJ(R)']),filtered_dataset['B:BQ5F(R)+ _E'] ,color = "orange")

plt.show()