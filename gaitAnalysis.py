import csv
import numpy as np
import matplotlib.pyplot as plt
import os

from numpy import genfromtxt
from matplotlib.backends.backend_pdf import PdfPages


# Moacir
#directory = '/home/maponti/Repos/mobile-fall-screening/data/'

#Patricia
#directory = 'C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\


def generatePdfFromCsv(directory = 'C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\'):

    """Function to generate Pdf from csv files"""

    j=1
    
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):
                f = directory+f
                data = genfromtxt(f, delimiter=',')
                lst = [elem for elem in data]
                N = len(lst)
                mat = np.bmat(lst)
                mat = np.reshape(mat,(N,4))
                plt.plot(mat[:,1])
                plt.plot(mat[:,2])
                plt.plot(mat[:,3])
                nome = "dados%03d.pdf"%j
                pp = PdfPages(nome)
                pp.savefig()
                pp.close()
                plt.clf()
                j=j+1


def createMatrixFromCsv(directory = 'C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\'):

    """Function to create matrix from csv data"""
    
    j = 1 
    dataAcc = [] # matriz vazia
        
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):
                f = directory + f
                data = genfromtxt(f, delimiter=',')
                lst = [elem for elem in data]
                N = len(lst)
                mat = np.bmat(lst)
                mat = np.reshape(mat,(N,4))
                jId=[j,'x'] # cria id com eixo x
                jId.append(mat[:,1]) # adiciona dados
                dataAcc.append(jId)
                jId=[j,'y'] # cria id com eixo y
                jId.append(mat[:,2]) # adiciona dados
                dataAcc.append(jId)
                jId=[j,'z'] # cria id com eixo z
                jId.append(mat[:,3]) # adiciona dados
                dataAcc.append(jId)
                
    return dataAcc                          
                                       


def espPot (directory = 'C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\'):

    """Function to generate pdf of the axes fusion, power spectrum and the logarithm of the power spectrum"""

    j=1

    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):

                # file name
                f = directory+f
                print(f)

                data = genfromtxt(f,delimiter=',')
                lst = [elem for elem in data]
                N = len(lst)
                mat = np.bmat(lst)
                mat = np.reshape(mat,(N,4))

                # signal fusion = sqrt( x^2 + y^2 + z^2 )
                matSquare = np.square(mat[:,1])+np.square(mat[:,2])+np.square(mat[:,3])
                matFusao = np.sqrt(matSquare)

                # convert array of arrays to single array
                matFusao = np.squeeze(np.asarray(matFusao[1:]))
                dataFftS = np.fft.fft(matFusao)
                espPot = np.abs(dataFftS)**2

                # number of frequencies to be displayed
                n = espPot.size
                nhalf = int(n/2)
                maxfr = 100
                timestep = 0.1
                freq = np.fft.fftfreq(n,timestep)

                fig = plt.figure()

                # signal fusion
                ax1 = fig.add_subplot(311)
                ax1.plot(matFusao)
                #ax1.set_title('Signal Fusion')

                # power spectrum
                ax2 = fig.add_subplot(312)
                ax2.plot(espPot[1:maxfr]+1)
                #ax2.set_title('Power Spectrum')

                # logarithm of the power spectrum
                ax3 = fig.add_subplot(313)
                ax3.plot(np.log(espPot[1:maxfr]+1))
                #ax3.set_title('Logarithm of the Power Spectrum')

                nome = "dados%03d.pdf"%j
                pp = PdfPages(nome)
                pp.savefig()
                pp.close()
                plt.show()
                plt.clf()
                j=j+1



def featuresAcc (directory = 'C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\'):

    """Function to ... """

    featMatrix = []

    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):

                # file name
                f = directory + f
                print(f)

                data = genfromtxt(f,delimiter=',')
                lst = [elem for elem in data]
                N = len(lst)
                mat = np.bmat(lst)
                mat = np.reshape(mat,(N,4))

                # signal fusion = sqrt( x^2 + y^2 + z^2 )
                matSquare = np.square(mat[:,1])+np.square(mat[:,2])+np.square(mat[:,3])
                matFusao = np.sqrt(matSquare)

                # convert array of arrays to single array
                matFusao = np.squeeze(np.asarray(matFusao[1:]))
                dataFftS = np.fft.fft(matFusao)
                espPot = np.abs(dataFftS)**2

                # number of frequencies to be displayed
                n = espPot.size
                nhalf = int(n/2)
                maxfr = 100
                timestep = 0.1
                freq = np.fft.fftfreq(n,timestep)

                espPotNyq = espPot[1:maxfr]

                #Power Spectral Entropy (represents a measure of energy compaction in transform coding)
                pse = sum(espPotNyq*np.log(espPotNyq)) 

                #Power Spectrum Peak(computed by finding the three highest values of signal)
                psp = np.max(espPotNyq)

                #Power Spectrum Peak Frequency (computed by finding the frequency related to the higher value of signal)
                pspf = np.argmax(espPotNyq)

                #Weighted Power Spectrum Peak (computed using the PSP values weighed by the PSPF values)
                wpsp = np.argmax(espPotNyq)*np.max(espPotNyq)


                print("Features:")
                print("PSE = " + str(pse))
                print("PSP = " + str(psp))
                print("PSPF = " + str(pspf))
                print("WPSP = " + str(wpsp))

                features = [pse, psp, pspf, wpsp]
                id_vol = "vol_%03d"%j
                months = -1

                featMatrix = [id_vol, features, months]
                
                
                #featMatrix.append([id_vol] + features + [months])
                
                #featMatrix = np.array(featMatrix)
                
 
    
    return featMatrix

                #### end of data reading


                # featMatrix will be a list

                #featMatrix = np.array(featMatrix)
                
                
                # show all features psp (column 2, index 1)   
                #print(featMatrix[:, 1])


