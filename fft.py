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

dataFftX = np.fft.fft(dataAcc[0]) # eixo X
dataFftY = np.fft.fft(dataAcc[1]) # eixo Y
dataFftZ = np.fft.fft(dataAcc[2]) # eixo Z


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

                                matSquare = np.square(mat[:,1])+np.square(mat[:,2])+np.square(mat[:,3])
                                matFusao = np.square(matSquare)

                                # opcao 1 - fft da fusao dos eixos
                                dataFttS = np.fft.fft(matFusao)
                                
                                # opcao 2 - fft de cada eixo separadamente
                                dataFttY = np.fft.fft(mat[:,1])
                                dataFttY = np.fft.fft(mat[:,2])
                                dataFttZ = np.fft.fft(mat[:,3])

                                # ???
                                dataFftArray.append(dataFftS)
        return dataFftArray




espPotX = (dataFftX.real + dataFftX.imag)**2
#espPot0 = np.reshape(espPot[0], (1, len(espPot[0]))) ## ???

plt.plot(espPotX)
plt.show()


# todos os dados
dataFFTteste = fft()

