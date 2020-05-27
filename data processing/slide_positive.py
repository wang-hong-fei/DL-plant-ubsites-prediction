# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:18:01 2019

@author: whfauto
"""

import pandas as pd
import numpy as np

df=pd.read_excel('plant.xlsx')
data=np.array(df)

filename=("positive.fasta")
num=0

for i in range(0,len(data)):
    position=data[i][2]
    seq=data[i][4]
    label='>' +data[i][1] + '_' +data[i][5]+ '_' + str(data[i][2]) 
    beg=position-16
    end=position+15
    target=seq[beg:end]
    if len(target)==31:
        with open(filename,'a') as f:
            f.write(label)
            f.write("\n")
            f.write(target)
            f.write("\n")
            num=num+1
            print (label,target)
            print (num)
    