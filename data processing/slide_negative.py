# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 09:54:08 2019

@author: whfauto
"""

import pandas as pd
import numpy as np

df=pd.read_excel('plant.xlsx')
data=np.array(df)

#print (data)

filename=("negative.fasta")
num=0
head=0
for i in range(0,len(data)):
    for j in range(head,data[i][2]-17):
        if data[i][4][j]=='K':
            position=j
            seq=data[i][4]
            label='>' +data[i][1] + '_' +data[i][5]+ '_' + str(j)
            beg=position-15
            end=position+16
            target=seq[beg:end]
            print (label)
            if len(target)==31:
                with open(filename,'a') as f:
                    f.write(label)
                    f.write("\n")
                    f.write(target)
                    f.write("\n")
    if i+1 != len(data) and data[i][0]==data[i+1][0]:
        #print (1)
        if data[i][2]+16<=data[i+1][2]-17:
            head=data[i][2]+16
            #print (head)
        else:
            #print (3)
            head=data[i+1][2]-17
    else:
        for k in range(data[i][2]+16,len(data[i][4])):
            if data[i][4][k]=='K':
                position=k
                seq=data[i][4]
                label='>' +data[i][1] + '_' +data[i][5]+ '_' + str(k)
                beg=position-15
                end=position+16
                target=seq[beg:end]
                print (label)
                #print (2)
                if len(target)==31:
                    with open(filename,'a') as f:
                        f.write(label)
                        f.write("\n")
                        f.write(target)
                        f.write("\n")
        head=0
                
#            num=num+1
#            print (num)
        

