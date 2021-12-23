import pandas as pd
import numpy as np
import math
from decimal import Decimal
import matplotlib.pyplot as plt

rescal = pd.read_csv("output/outputRESCAL.csv")
distmult = pd.read_csv("output/outputDistMult.csv")
complex = pd.read_csv("output/outputComplEx.csv")
transe = pd.read_csv("output/outputTransE.csv")

rescalUnF = rescal.iloc[4:8,1]
distmultUnF = distmult.iloc[4:8,1]
complexUnF = complex.iloc[4:8,1]
transeUnF = transe.iloc[4:8,1]


rescalF = rescal.iloc[[41,32,23,14], 1]
rescalF = rescalF.reset_index(drop=True)
distmultF = distmult.iloc[[11,10,9,8],1]
distmultF = distmultF.reset_index(drop=True)
complexF = complex.iloc[[11,10,9,8],1]
complexF = complexF.reset_index(drop=True)
transeF = transe.iloc[[11,10,9,8],1]
transeF = transeF.reset_index(drop=True)

HitsUnF = pd.DataFrame({'RESCAL':rescalUnF, 'DistMult':distmultUnF, 'ComplEx':complexUnF, 'TransE':transeUnF})
HitsF = pd.DataFrame({'RESCAL':rescalF, 'DistMult':distmultF, 'ComplEx':complexF, 'TransE':transeF})
print(HitsUnF)
print(HitsF)
x = [1,2,3,4]
ticks = [1,2,3,4]
labels = ['@1', '@10', '@100', '@1000']

def round_sig(x, sig=3):
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

for k in range(len(HitsUnF)):
    for i in range(len(HitsUnF.columns)):
        HitsUnF.iloc[k,i] = Decimal(HitsUnF.iloc[k,i])
        HitsUnF.iloc[k,i] = round_sig(HitsUnF.iloc[k,i])
        HitsF.iloc[k,i] = Decimal(HitsF.iloc[k,i])
        HitsF.iloc[k,i] = round_sig(HitsF.iloc[k,i])


plt.figure()
plt.plot(x, HitsUnF.iloc[:,0], 'black', label='RESCAL')
plt.plot(x, HitsUnF.iloc[:,1], 'red', label='DistMult')
plt.plot(x, HitsUnF.iloc[:,2], 'green', label='ComplEx')
plt.plot(x, HitsUnF.iloc[:,3], 'blue', label='TransE')
plt.xticks(ticks,labels)
plt.legend(loc='best')
plt.title('Unfiltered hits')
plt.ylabel('Hit Percentage')
plt.ylim(0,1)
plt.show()


plt.figure()
plt.plot(x, HitsF.iloc[:,0], 'black', label='RESCAL')
plt.plot(x, HitsF.iloc[:,1], 'red', label='DistMult')
plt.plot(x, HitsF.iloc[:,2], 'green', label='ComplEx')
plt.plot(x, HitsF.iloc[:,3], 'blue', label='TransE')
plt.xticks(ticks,labels)
plt.legend(loc='best')
plt.title('Filtered hits')
plt.ylabel('Hit Percentage')
plt.ylim(0,1)
plt.show()


