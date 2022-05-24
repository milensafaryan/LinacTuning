import sys
import ROOT
import pandas as pd

date = sys.argv[1]
which = sys.argv[2]
filename ='../data/%s/devicescan_%s.csv'%(date,which)

df = pd.read_csv(r'%s'%filename)
df.drop(list(df.filter(regex = 'Timestamp|iteration')), axis = 1, inplace = True)
lms = list(df.filter(regex='LM'))
df['LMSM'] = df[lms].sum(axis=1)
data = {key: df[key].values for key in list(df.keys())}

rdf = ROOT.RDF.MakeNumpyDataFrame(data)

rdf.Display().Print()

outname = '%s_%s.root'%(which,date)
rdf.Snapshot('paramT',outname)


