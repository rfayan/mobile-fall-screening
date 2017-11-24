import csv
import numpy as np
import matplotlib.pyplot as plt
import os


from numpy import genfromtxt
from matplotlib.backends.backend_pdf import PdfPages


def logEspPot (directory = 'C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\'):
    j=1
    
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):
                f=directory+f
                data=genfromtxt(f,delimiter=',')
                lst=[elem for elem in data]
                N= len(lst)
                mat = np.bmat(lst)
                mat = np.reshape(mat,(N,4))
                matSquare = np.square(mat[:,1])+np.square(mat[:,2])+np.square(mat[:,3])
                matFusao = np.square(matSquare)
                dataFftS= np.fft.fft(matFusao)
                espPot=(dataFftS.real+dataFftS.imag)**2
                n=espPot.size
                timestep=0.1
                freq=np.fft.fftfreq(n,timestep)
                logEspPot= plt.plot(freq, np.log(espPot+1))
                nome="dados%03d.pdf"%j
                pp = PdfPages(nome)
                pp.savefig()
                pp.close()
                plt.clf()
                j=j+1
