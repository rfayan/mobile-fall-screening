import csv
import numpy as np
import matplotlib.pyplot as plt
import os

from numpy import genfromtxt
from matplotlib.backends.backend_pdf import PdfPages

# Moacir
#def espPot (directory = '/home/maponti/Repos/mobile-fall-screening/data/'):
#directory = '/home/maponti/Repos/mobile-fall-screening/data/'

data = genfromtxt ('C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\010_Accelerometer_20170724-141126735.csv', delimiter=',')
j = 1

lst = [elem for elem in data]
N= len(lst)
mat = np.bmat(lst)
mat = np.reshape(mat,(N,4))

# signal fusion = sqrt( x^2 + y^2 + z^2 )
matSquare = np.square(mat[:,1])+np.square(mat[:,2])+np.square(mat[:,3])
matFusao = np.sqrt(matSquare)

## converte array de arrays em um unico array
matFusao = np.squeeze(np.asarray(matFusao[1:]))

dataFftS= np.fft.fft(matFusao)
espPot = np.abs(dataFftS)**2

# numero de frequencias a ser exibida
n = espPot.size
nhalf = int(n/2)
maxfr = 100
timestep=0.1
freq=np.fft.fftfreq(n,timestep)

fig = plt.figure()

# sinal
ax1 = fig.add_subplot(311)
ax1.plot(matFusao)
#ax1.set_title('Signal Fusion')


# espectro de potencia
ax2 = fig.add_subplot(312)
ax2.plot(espPot[1:maxfr]+1)
#ax2.set_title('Power Spectrum')


# log do espectro
ax3 = fig.add_subplot(313)
ax3.plot(np.log(espPot[1:maxfr]+1))
#ax3.set_title('Logarithm of the Power Spectrum')


nome="dados%03d.pdf"%j
pp = PdfPages(nome)
pp.savefig()
pp.close()
plt.show()
plt.clf()
j=j+1

#Power Spectral Entropy (represents a measure of energy compaction in transform coding)
pse = sum(espPot*np.log(espPot)) 

#Power Spectrum Peak(computed by finding the three highest values of signal)
psp = np.max(freq)

#Power Spectrum Peak Frequency (computed by finding the frequency related to the higher value of signal)
pspf = np.argmax(espPot)

#Weighted Power Spectrum Peak (computed using the PSP values weighed by the PSPF values)
wpsp = np.argmax(espPot)**np.max(freq)

 


            
