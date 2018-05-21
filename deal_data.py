import pandas as pd
import numpy as np


df = pd.read_excel("creditdata.xlsx")
data_1=df[0:284261].as_matrix()[:,2:]
data_2=df[284262:284727].as_matrix()[:,2:]
data_3=df[284727:284807].as_matrix()[:,2:]

np.save('data_1.npy',data_1)
np.save('data_2.npy',data_2)
np.save('data_3.npy',data_3)
'''
a=np.ones((2,3))
print(a)
np.concatenate((a,a))
print(a)
print(np.shape(a))
2-284262
284263-284728
284729-284808
'''



#df.head()