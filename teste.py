import csv
import numpy as np
import matplotlib.pyplot as plt
import os

from numpy import genfromtxt

data=genfromtxt('C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\001_Accelerometer_20170719-161640829.csv', delimiter=',')

data = data[1:,:]

lst = [elem for elem in data]
N = len (lst)
mat = np.bmat (lst)
mat = np.reshape(mat,(N,4))
dataAcc=[]

dataAcc.append(mat[:,1])
dataAcc.append(mat[:,2])
dataAcc.append(mat[:,3])

dataFftX = np.fft.fft(dataAcc[0]) # eixo X
dataFftY = np.fft.fft(dataAcc[1]) # eixo Y
dataFftZ = np.fft.fft(dataAcc[2]) # eixo Z

matSquare=np.square(mat[:,1])+np.square(mat[:,2])+np.square(mat[:,3]) 
matFusao=np.square(matSquare) # fusao

fftFusao=np.fft.fft(matSquare) # fft fusao

espPot= (fftFusao.real + fftFusao.imag)**2  # espectro de potencia original

n = espPot.size
timestep = 0.1
freq = np.fft.fftfreq(n,timestep)
plt.plot(freq,espPot)

logEspPot= plt.plot(freq,np.log(espPot+1))# logaritmo do espectro de potencia


psp = np.argmax(espPot) # PSP

wpsp = np.argmax(espPot)*np.max(espPot) # WPSP

pse = sum (espPot * np.log(espPot)) #PSE

