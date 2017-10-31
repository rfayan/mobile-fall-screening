import csv
import numpy as np
import matplotlib.pyplot as plt
import os

from numpy import genfromtxt


data = genfromtxt('C:\\Users\\patricia\\Desktop\\Dados Acelerômetro\\001_Accelerometer_20170719-161640829.csv', delimiter=',')
data = data[1:,:]

lst = [elem for elem in data]
N = len (lst)
mat = np.bmat (lst)
mat = np.reshape(mat,(N,4))
dataAcc=[]

dataAcc.append(mat[:,1])
dataAcc.append(mat[:,2])
dataAcc.append(mat[:,3])

dataFft = np.fft.fft(dataAcc)


def fft(directory = 'C:\\Users\\patricia\\Desktop\\Dados Acelerômetro\\'):
        dataFftArray = []
        for root, dirs, files in os.walk(directory):
                for f in files:
                        if f.endswith(".csv"):
                                f = directory+f
                                data = genfromtxt(f, delimiter=',')
                                data = data[1:,:]
                                lst = [elem for elem in data]
                                N = len(lst)
                                mat = np.bmat(lst)
                                mat = np.reshape(mat,(N,4))
                                dataAcc = [] 
                                dataAcc.append(mat[:,1])
                                dataAcc.append(mat[:,2])
                                dataAcc.append(mat[:,3])
                                dataFtt = np.fft.fft(dataAcc)
                                dataFftArray.append(dataFft)                                     
        return dataFftArray



espPot = (dataFft.real + dataFft.imag)**2
plt.plot(espPot[0])
plt.plot(espPot[1]) 
plt.plot(espPot[2])
# ....


