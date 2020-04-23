import pandas as pd

df1 = pd.read_csv('../shared/BaseFilecsv.csv')
df2 = pd.read_csv('../shared/BaseFileResultcsv.csv')

df3 = pd.merge(df1,df2,on=['gvkey','fyear'],how="outer")
#print(df3)
df3.to_csv('../shared/WithStopWordscsv.csv',index=False,sep=',')