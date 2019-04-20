import pickle
import os
data=[]
for file in sorted(os.listdir('./rawdata')):
    d1=pickle.load(open( './rawdata/'+file, "rb" ))
    print(len(d1))
    data.append(d1)
print(len(data))
