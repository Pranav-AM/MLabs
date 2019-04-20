#!/usr/bin/env python
# coding: utf-8


import pickle
from pprint import pprint
obj = ''

with open('../../../data/persondata.pkl', 'rb') as f:
    obj = pickle.load(f)

print "Total pose estimation annotations: ", len(obj)

#for ob in obj:
#    pprint(ob)

#print(len(obj[0]))

imglist = []
for idx, item in enumerate(obj):
    imgname = item[0].split('.')[0]
    
    pername = item[1]
    bbox = item[2]
    jointloc = item[3]
    detailspath = "persondata/" + imgname + ".txt"
    appstr = item[1] + " " + str(bbox[0][0]) + " " + str(bbox[0][1]) + " " + str(bbox[2][0]) + " " + str(bbox[2][1]) +" ";
    count = 0
    
    
        
    for it in jointloc:
        if count > it[1]:
            continue
        while count != it[1]:
            #print it[1] , count
            appstr += "-1 -1 "
            count += 1
        appstr += str(it[0][0]) + " " + str(it[0][1]) + " "
        count +=1
    while count != 18:
        appstr += "-1 -1 "
        count += 1

    appstr += "\n"

    # If the program is opening the detailspath file for the first time, write into it
    # otherwise append to it
    if(imgname in imglist):
    	f = open(detailspath, "a+")
    else:
	imglist.append(imgname)
        f = open(detailspath, "w+")

    f.write(appstr) 
    #f.close()


'''
lst = sorted(imglist)
for i in range(len(lst)-1):
    if lst[i] == lst[i+1]:
        print(lst[i])
'''





