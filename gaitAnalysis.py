import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import regex as re
import scipy
import scipy.fftpack
import pylab

from scipy import pi
from scipy import stats
from scipy.signal import butter, lfilter, freqz
from numpy import genfromtxt
from matplotlib.backends.backend_pdf import PdfPages


# Moacir
directory = '/home/maponti/Repos/mobile-fall-screening/data/'

# listas
index_faller = []
index_excluded = []

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
                                       


def espPot (directory = 'C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\', filtering=False):

    """Function to generate pdf of the axes fusion, power spectrum and the logarithm of the power spectrum"""

    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):

                # file name
                f = directory+f
                print(f)

                # pega o ultimo elemento apos particionar com \ ou /
                # desse pega os tres primeiros valores
                # e depois converte para inteiro
                j = int(re.split('\\ |/', f)[-1][0:3]) # Linux
                #j = int(re.split('\\\\', f)[-1][0:3]) # Windows

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
                espPotS = np.abs(dataFftS)**2

                # filtering
                if filtering == True:
                    fs = 50
                    matFusaoF = butter_lowpass_filter(matFusao, 3.6667, fs, order=6)
                    dataFftF = np.fft.fft(matFusaoF)
                    espPotF = np.abs(dataFftF)**2

                # number of frequencies to be displayed
                n = espPotS.size
                nhalf = int(n/2)
                maxfr = 100
                timestep = 0.1
                freq = np.fft.fftfreq(n,timestep)

                fig = plt.figure()

                # signal fusion
                ax1 = fig.add_subplot(311)
                ax1.plot(matFusao)
                if filtering:
                    ax1.plot(matFusaoF)
                #ax1.set_title('Signal Fusion')

                # power spectrum
                ax2 = fig.add_subplot(312)
                ax2.plot(espPotS[1:maxfr]+1)
                if filtering:
                    ax2.plot(espPotF[1:maxfr]+1)
                #ax2.set_title('Power Spectrum')

                # logarithm of the power spectrum
                ax3 = fig.add_subplot(313)
                ax3.plot(np.log(espPotS[1:maxfr]+1))
                if filtering:
                   ax3.plot(np.log(espPotF[1:maxfr]+1))
                #ax3.set_title('Logarithm of the Power Spectrum')

                nome = "dados%03d.pdf"%j
                pp = PdfPages(nome)
                pp.savefig()
                pp.close()
                #plt.show()
                #plt.clf()



def featuresAcc (directory = 'C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\', filtering=False):

    """Function to ... """

    featMatrix = []

    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith(".csv"):

                # file name
                f = directory + f
                print(f)

                # pega o ultimo elemento apos particionar com \ ou /
                # desse pega os tres primeiros valores
                # e depois converte para inteiro
                j = int(re.split('\\ |/', f)[-1][0:3])  # Linux
                #j = int(re.split('\\\\', f)[-1][0:3]) # Windows

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
                dataFftS = np.fft.fft(matFusao[5:])
                espPot = np.abs(dataFftS)**2

                # filtering
                if filtering == True:
                    fs = 25
                    matFusaoF = butter_lowpass_filter(matFusao, 3.667 , fs, order=6)
                    dataFftF = np.fft.fft(matFusaoF)
                    espPotF = np.abs(dataFftF)**2

                # number of frequencies to be displayed
                n = espPot.size
                nhalf = int(n/2)
                maxfr = 50
                timestep = 0.1
                freq = np.fft.fftfreq(n,timestep)

                espPotNyq = espPot[1:maxfr]

                #Power Spectral Entropy (represents a measure of energy compaction in transform coding)
                pse = sum(espPotNyq*np.log(espPotNyq)) 

                #Power Spectrum Peak(computed by finding the three highest values of signal)
                psp = np.max(espPotNyq)

                #Power Spectrum Peak Frequency (computed by finding the frequency related to the higher value of signal)
                pspf1 = np.argmax(espPotNyq)

                #Weighted Power Spectrum Peak (computed using the PSP values weighed by the PSPF values)
                wpsp = np.argmax(espPotNyq)*np.max(espPotNyq)

                # get second peak
                espPotNyq[pspf1] = 0
                espPotNyq[pspf1-1] = 0
                espPotNyq[pspf1+1] = 0
                #Power Spectrum Peak Frequency (computed by finding the frequency related to the higher value of signal)
                pspf2 = np.argmax(espPotNyq)

                # get third peak
                espPotNyq[pspf2] = 0
                espPotNyq[pspf2-1] = 0
                espPotNyq[pspf2+1] = 0
                #Power Spectrum Peak Frequency (computed by finding the frequency related to the higher value of signal)
                pspf3 = np.argmax(espPotNyq)

                print("Features:")
                print("PSE = " + str(pse))
                print("PSP = " + str(psp))
                print("PSPF = " + str(pspf1) + " " + str(pspf2) + " " + str(pspf3))
                print("WPSP = " + str(wpsp))

                features = [pse, psp, pspf1, pspf2, pspf3, wpsp]
                id_vol = "%03d"%j
                months = -1
                                
                featMatrix.append([id_vol] + features + [months])

                
    #featMatrix = np.array(featMatrix)
    
    return featMatrix



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



def setMonths(matrix, posInd, value):
    "Percorre a matriz, e altera as posicoes de forma que o mes fique 'value'"
    " posInd contem as posicoes a serem alteradas para 'value' "
    " ex. setMonths(mat, [1, 5, 6, 67], 1) "
    " ex. setMonths(mat, lista_indices, 0) "
     
    print(posInd)




def tTestFeatures(matriz, featId):

    mPos = []
    mNeg = []

    # percorre todos as linhas
    for v in matriz:
        # monta as matrizes negativa e positiva
        if v[5] == 1:
            mPos = mPos + [v[featId]]
        elif v[5] == -1:
            mNeg = mNeg + [v[featId]]

    print("Medias, grupo + : " + str(np.mean(mPos)) + " desvio: " + str(np.std(mPos)))
    print("        grupo - : " + str(np.mean(mNeg)) + " desvio: " + str(np.std(mNeg)))


    tTest = stats.ttest_ind(mPos, mNeg) 
    print("p-value Test-t: "+ str(tTest.pvalue))

    return tTest
     




