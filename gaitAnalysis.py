import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import regex as re
import scipy
import scipy.fftpack
import pylab
import heapq
import pickle as pkl
import pandas as pd
import copy

from scipy import pi
from scipy import stats
from scipy.signal import butter, lfilter, freqz
from numpy import genfromtxt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product


# Moacir
#directory = '/home/maponti/Repos/mobile-fall-screening/data/'

#Patricia
directory = 'C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\'

#lista de idades(grupos)
index_60 = [0, 6, 8, 9, 10, 11, 13, 15, 18, 20, 23, 24, 25, 27, 28, 31, 33, 34, 35, 36, 37, 41, 43, 45, 46, 47, 49, 50, 52, 54, 55, 60, 63, 65, 69, 70, 72, 74, 75, 76]
index_70 = [1, 3, 4, 7, 14, 17, 21, 22, 32, 38, 39, 40, 42, 53, 56, 57, 58, 61, 62, 64, 71, 77]
index_80 = [2, 12, 19, 29, 30, 44, 51, 59, 67, 73, 78]


# listas de caidores e excluidos
index_faller = [7, 9, 10, 15, 27, 34, 35, 40, 45, 58, 59, 63, 70, 77]
index_faller3M = [9, 10, 35, 40, 59, 70, 77]
index_faller6M = [7, 15, 27, 34, 46, 58, 59, 63]
index_excluded = [5, 16, 26, 48, 65, 66]

# dicionario com indices e meses
# podemos usar meses, e depois se precisar convertemos tudo para '1' (caidor)
dict_label = {7:6, 15:6, 27:6, 34:6, 46:6, 58:6, 59:6, 63:6, 9:3, 10:3, 35:3, 40:3, 59:3, 70:3, 77:3}

# dicionario com o numero de quedas
dict_qtde  = {7:1, 9:1, 10:1, 15:1, 27:1, 34:1, 35:1, 40:1, 45:1, 58:1, 59:2, 63:1, 70:1, 77:1}


def generatePdfFromCsv(directory):

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


def createMatrixFromCsv(directory):

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
                                       

def zscore_normalization(signal):
    return (signal-np.mean(signal))/np.std(signal)

def espPot (directory, filtering=False):

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


def segmentTS_TUG(tug, sumFilterSize=300):
    # filtro para realizar convolucao
    # h = [1, 1, 1]
    h = np.ones(sumFilterSize)
    matSegm = np.convolve(tug, h, mode='same')

    # normalizacao z-score
    matSegm = (matSegm-np.mean(matSegm)) / np.std(matSegm)

    matSegm[ : int(sumFilterSize/2)] = 0
    matSegm[-int(sumFilterSize/2) : ] = 0

    # especificar um limiar e pegar apenas valores acima
    mask = np.zeros(matSegm.shape)
    mask[np.where(matSegm > 0)] = 1

    # segunda segmentacao (em cima da mascara anterior)
    matSegm2 = np.convolve(mask, h, mode='same')
    matSegm2 = (matSegm2-np.mean(matSegm2)) / np.std(matSegm2)

    matSegm2[ : int(sumFilterSize/2)] = 0
    matSegm2[-int(sumFilterSize/2) : ] = 0

    # especificar um limiar e pegar apenas valores acima
    mask2 = np.zeros(matSegm2.shape)
    mask2[np.where(matSegm2 > 0)] = 1

    # filtro
    size = 100 #trocar aqui
    for filtro in range(mask.size):
        accumulator = 0
        interior = 0
        while(interior <= size):
            auxiliary = size+interior
            if(auxiliary >= mask.size):
                auxiliary = mask.size-1
            accumulator += mask[auxiliary]            
            interior += 1
        if(mask[size]==0 and accumulator > 0):
            mask[size]=1
        if(mask[size]==1 and accumulator == 0):
            mask[size]=0  

    print(mask)
    #### segmentacao eh ate aqui

    return mask2, mask
    

def segmentTUGs(directory, filtering=False, sumFilterSize=300):

    ''' Function to segment the different TUGs along the signal '''
    ''' sumFilterSize default 300 = 3 seconds (considering 100Hz sampling)  '''
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):

                # file name
                f = directory + f
                print(f)

                # pega o ultimo elemento apos particionar com \ ou /
                # desse pega os tres primeiros valores
                # e depois converte para inteiro
                #j = int(re.split('\\ |/', f)[-1][0:3]) # Linux
                j = int(re.split('\\\\', f)[-1][0:3]) # Windows

                data = genfromtxt(f, delimiter=',')
                lst = [elem for elem in data]
                N = len(lst)
                mat = np.bmat(lst)
                mat = np.reshape(mat,(N,4))

                # signal fusion = sqrt( x^2 + y^2 + z^2 )
                matSquare = np.square(mat[:,1])+np.square(mat[:,2])+np.square(mat[:,3])
                matFusao = np.sqrt(matSquare)

                # convert array of arrays to single array
                matFusao = np.squeeze(np.asarray(matFusao[1:]))
                matFusao[0:5] = 1

                mask2, mask = segmentTS_TUG(matFusao, sumFilterSize=sumFilterSize)

                dataFftS = np.fft.fft(matFusao)
                espPotS = np.abs(dataFftS)**2

                # filtering
                if filtering == True:
                    fs = 50
                    matFusaoF = butter_lowpass_filter(matFusao, 3.6667, fs, order=6)
                    dataFftF = np.fft.fft(matFusaoF)
                    espPotF = np.abs(dataFftF)**2

                fig = plt.figure()

                # signal fusion
                ax1 = fig.add_subplot(211)
                ax1.plot(matFusao)
                if filtering:
                    ax1.plot(matFusaoF*mask)
                    ax1.plot(matFusaoF*mask2)
                #ax1.set_title('Signal Fusion')

                # segmentation
                ax2 = fig.add_subplot(212)
                #ax2.plot(matSegm)
                ax2.plot(mask)
                ax2.plot(mask2)
                nome = "dados_segmentacao%03d.pdf"%j
                pp = PdfPages(nome)
                pp.savefig()
                pp.close()
                #plt.show()
                plt.clf()
                plt.close('all')

                if j == 5:
                    return mask2, matFusao



def countTUGs(series):
    ''' Counts the number of connected sequences of -1s inside
        the mask segmenting TUGs
        Returns:
            count: the number of TUGs detected by segmentation
            TUGsizes: a vector of 'count' elements, each with the
                    number of observations for each segmented TUG
            TUGpos: a vector of 'count' elements, each with the
                    starting position of each segmented TUG
    '''
    count = 0
    curr = 0
    mask = series
    
    TUGpos = []

    # encontra primeiro '1'
    one  = np.argmax(series[curr:])
    TUGpos.append(one) # guarda a primeira posicao

    curr = curr + one
    j = 1

    TUGsizes = []

    while (one > 0):
        #encontra o proximo '0'
        zero = np.argmin(series[curr:])
        print("TUG_%03d:"%j + str(zero))
        TUGsizes.append(zero)
        curr = curr + zero
        count = count + 1
        
        # encontra o proximo '1'
        one  = np.argmax(series[curr:])
        curr = curr + one
        TUGpos.append(curr)

        j = j + 1

    del TUGpos[-1] # remove last position

    return count, TUGsizes, TUGpos


def agroup(count, mask, tugs, min_size = 400, max_size=1000):
    ''' checks the size of each TUG (observations)
        in order to split those above some max threshold (max_size)
        and merge those below some minimum threshold (min_size)
    '''

    countTUG = count[0]  # number of TUGs
    TUGsizes = count[1]  # size of each TUG
    TUGpos   = count[2]  # starting position of each TUG

    # for each TUG size
    for j in range(len(TUGsizes)):
        pos = TUGpos[j]
        size= TUGsizes[j]
        # split
        if (size > max_size):
           # new size for the first part
            size = int(size/2)
            mask[pos+size] = 0

            count, TUGsizes, TUGPos = countTUGs(mask) 

        #merge
        #elif (size < min_size):
    
    newCount = len(TUGsizes)   
    return newCount, TUGsizes, mask


def featuresAcc (directory, filtering=False):

    """Function to generate signal characteristics"""

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
                #j = int(re.split('\\ |/', f)[-1][0:3])  # Linux
                j = int(re.split('\\\\', f)[-1][0:3]) # Windows

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

                #power spectral entropy
                pse = sum(espPotNyq*np.log(0.0001 + espPotNyq)) 

                #power spectrum peak
                psp1 = np.max(espPotNyq)

                #power spectrum peak frequency
                pspf1 = np.argmax(espPotNyq)

                #weighted power spectrum peak
                wpsp = np.argmax(espPotNyq)*np.max(espPotNyq)

                # get second peak pspf e psp
                espPotNyq[pspf1] = 0
                espPotNyq[pspf1-1] = 0
                espPotNyq[pspf1+1] = 0
                
                pspf2 = np.argmax(espPotNyq)
                psp2  = np.max(espPotNyq)

                # get third peak pspf e psp
                espPotNyq[pspf2] = 0
                espPotNyq[pspf2-1] = 0
                espPotNyq[pspf2+1] = 0

                pspf3 = np.argmax(espPotNyq)
                psp3  = np.max(espPotNyq)

                print("Features:")
                print("PSE = " + str(pse))
                print("PSP = " + str(psp1) + " " + str(psp2) + " " + str(psp3))
                print("PSPF = " + str(pspf1) + " " + str(pspf2) + " " + str(pspf3))
                print("WPSP = " + str(wpsp))

                features = [pse, psp1, psp2, psp3, pspf1, pspf2, pspf3, wpsp]
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



def tTestFeatures(matrix, featId, indPos, indExc):
    
    """Function for the variables statistical analysis of the two 
    groups formed from the two months of future fall observation 
    using test t
    
    matrix - the matrix with rows representing subjects
             first position is the ID, last position is the label
    featId - the feature to be tested ( > 0 )
    
    indPos - indices of fallers

    indExc - indices of subject to be excluded
    
    """

    labPos = 9 # index of the label in the feature vector
        
    mPos = []
    mNeg = []

    #rotula na matriz
    for i in indPos:
        matrix[i][labPos] = 1   

    for i in indExc:
        matrix[i][labPos] = 0   

    # percorre todos as linhas
    for v in matrix:
        # monta as matrizes negativa e positiva
        if v[labPos] == 1:
            mPos = mPos + [v[featId]]
        elif v[labPos] == -1:
            mNeg = mNeg + [v[featId]]
    
    print("Feature %d" % (featId))
    print("\tMedias, grupo +: %02.4f, desvio: %.4f" %(np.mean(mPos), np.std(mPos)))
    print("\tMedias, grupo -: %02.4f, desvio: %.4f" %(np.mean(mNeg), np.std(mNeg)))


    tTest = stats.ttest_ind(mPos, mNeg) 
    print("\tp-value t test: %.4f " % (tTest.pvalue))

    return tTest


def anovaGroups(matrix, featId, group1, group2, group3):

    """Function for the variables statistical analysis of the three 
    initial groups using anova"""
    
    labPos = 9

    m60 = []
    m70 = []
    m80 = []

    # copy matrix
    tmatrix = np.zeros(matrix.shape)
    tmatrix[:] = matrix

    #rotula na matriz
    for i in group1:
        tmatrix[i][labPos] = 1

    for i in group2:    
        tmatrix[i][labPos] = 2

    for i in group3:
        tmatrix[i][labPos] = 3

    #percorre todas as linhas
    for v in tmatrix:
        #monta as matrizes dos grupos
        if v[9] == 1:
            m60 = m60 + [v[featId]]
        elif v[9] == 2:
            m70 = m70 + [v[featId]]
        else: 
            m80 = m80 + [v[featId]]

    print("Feature %d" % (featId))
    print("\tMedias, grupo 60-69: %02.4f, desvio: %.4f" %(np.mean(m60), np.std(m60)))
    print("\tMedias, grupo 70-79: %02.4f, desvio: %.4f" %(np.mean(m70), np.std(m70)))
    print("\tMedias, grupo 80+  : %02.4f, desvio: %.4f" %(np.mean(m80), np.std(m80)))
    anova = stats.f_oneway(m60, m70, m80)
    print("\tp-value anova: %.4f " % (anova.pvalue))

    return anova

 


#########################  Refactoring

def data_read_csv(directory, so="Windows", savefile=True, filename="data_accelerometer.pkl"):

    """Function to read csv files""" 

    data = []
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith(".csv"):

                # file name
                f = directory + f
                print(f)

                # pega o ultimo elemento apos particionar com \ ou /
                # desse pega os tres primeiros valores
                # e depois converte para inteiro
                if so == "Windows":
                    j = int(re.split('\\\\', f)[-1][0:3]) # Windows
                elif so == "Linux":
                    j = int(re.split('\\ |/', f)[-1][0:3]) # Linux

                dataf= genfromtxt(f, delimiter=',')
                lst = [elem for elem in dataf]
                N = len(lst)
                mat = np.bmat(lst)
                mat = np.reshape(mat,(N,4))


                jId=[j,'x'] # cria id com eixo x
                jId.append(np.squeeze(np.asarray(mat[1:,1]))) # adiciona dados x
                data.append(jId)
                jId=[j,'y'] # cria id com eixo y
                jId.append(np.squeeze(np.asarray(mat[1:,2]))) # adiciona dados y
                data.append(jId)
                jId=[j,'z'] # cria id com eixo z
                jId.append(np.squeeze(np.asarray(mat[1:,3]))) # adiciona dados z
                data.append(jId)
              
    data = pd.DataFrame(data)

    # saves the read data into a file               
    if savefile:
        data.to_pickle(filename)

    return data


def read_data_pkl(filename="data_accelerometer.pkl"):
    return pd.read_pickle(filename)



##fusion from data_read_csv

def data_fusion(data, savefile=True, filename="data_fusion.npy"):
    
   
    """Function to generate the axes fusion"""
 
    #get maximum number
    N = max(data[0])

    dataf = []

    for i in range(N):

        st = (i*3)
        datai = data.loc[st:st+2]
        print(datai)
        # signal fusion = sqrt( x^2 + y^2 + z^2 )
        x = np.asarray(datai[2][st])
        y = np.asarray(datai[2][st+1])
        z = np.asarray(datai[2][st+2])
        fusion = np.sqrt(np.square(x)+np.square(y)+np.square(z))
        # convert array of arrays to single array
        #fusion = np.squeeze(np.asarray(matFusao[1:]))

        dataf.append(fusion)
        print(fusion)
 
    if savefile:
        np.save(filename, dataf)

    return dataf


def read_data_npy(filename="data_fusion.pkl"):
    return np.load(filename)




## features from data_fusion

def data_features(data, filtering=True, savefile=True, filename="featuresAcc.npy"):

    """Function to generate signal characteristics"""

    featMatrix = []
    j = 1

    for i in data:
        dataFftS = np.fft.fft(i)
        espPot = np.abs(dataFftS)**2

        # filtering
        if filtering == True:
            fs = 25
            dataF = butter_lowpass_filter(i, 3.667 , fs, order=6)
            dataFftF = np.fft.fft(dataF)
            espPotF = np.abs(dataFftF)**2

        # number of frequencies to be displayed
        n = espPot.size
        nhalf = int(n/2)
        maxfr = 50
        timestep = 0.1
        freq = np.fft.fftfreq(n,timestep)

        espPotNyq = espPot[1:maxfr]

        #power spectral entropy
        #computed over the Spectrum normalized between 0-1
        espProb = espPotNyq.copy()
        espProb = (espProb - np.min(espProb)) / (np.max(espProb) - np.min(espProb))
        pse = -sum(espProb*np.log(0.0001 + espProb)) 

        # the remaining features are computed over a
        # z-score normalization (i.e. x-mean/sd)
        espPotNyq = (espPotNyq-np.mean(espPotNyq)) / np.std(espPotNyq)

        #power spectrum peak
        psp1 = np.max(espPotNyq)

        #power spectrum peak frequency
        pspf1 = np.argmax(espPotNyq)

        #weighted power spectrum peak
        wpsp = np.argmax(espPotNyq)*np.max(espPotNyq)

        # get second peak pspf e psp
        espPotNyq[pspf1] = 0
        espPotNyq[pspf1-1] = 0
        espPotNyq[pspf1+1] = 0
                
        pspf2 = np.argmax(espPotNyq)
        psp2  = np.max(espPotNyq)

        # get third peak pspf e psp
        espPotNyq[pspf2] = 0
        espPotNyq[pspf2-1] = 0
        espPotNyq[pspf2+1] = 0

        pspf3 = np.argmax(espPotNyq)
        psp3  = np.max(espPotNyq)

        print("Features:")
        print("PSE = " + str(pse))
        print("PSP = " + str(psp1) + " " + str(psp2) + " " + str(psp3))
        print("PSPF = " + str(pspf1) + " " + str(pspf2) + " " + str(pspf3))
        print("WPSP = " + str(wpsp))

        features = [pse, psp1, psp2, psp3, pspf1, pspf2, pspf3, wpsp]
        id_vol = "%03d"%j
        months = -1
        j = j + 1
        featMatrix.append([id_vol] + features + [months])

    if savefile:
        np.save(filename, featMatrix)

    return featMatrix


def read_featuresAcc_npy(filename="featureAcc.pkl"):
    return np.load(filename)



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y




## segmentation from data_fusion

def data_segmentTS_TUG(tug, sumFilterSize=300, savefile=True, filename="segmentation.pkl"):
    
    maskM = []
    segmT = []

    for i in tug:
        # filtro para realizar convolucao
        # h = [1, 1, 1]
        h = np.ones(sumFilterSize)
        matSegm = np.convolve(i, h, mode='same')

        # normalizacao z-score
        matSegm = zscore_normalization(matSegm)

        matSegm[ : int(sumFilterSize/2)] = 0
        matSegm[-int(sumFilterSize/2) : ] = 0

        # especificar um limiar e pegar apenas valores acima
        mask = np.zeros(matSegm.shape)
        mask[np.where(matSegm > 0)] = 1

        # segunda segmentacao (em cima da mascara anterior)
        matSegm2 = np.convolve(mask, h, mode='same')
        matSegm2 = zscore_normalization(matSegm2)

        matSegm2[ : int(sumFilterSize/2)] = 0
        matSegm2[-int(sumFilterSize/2) : ] = 0

        # especificar um limiar e pegar apenas valores acima
        mask2 = np.zeros(matSegm2.shape)
        mask2[np.where(matSegm2 > 0)] = 1

        # filtro
        size = 100 #trocar aqui
        for filtro in range(mask.size):
            accumulator = 0
            interior = 0
            while(interior <= size):
                auxiliary = size+interior
                if(auxiliary >= mask.size):
                    auxiliary = mask.size-1
                accumulator += mask[auxiliary]            
                interior += 1
            if(mask[size]==0 and accumulator > 0):
                mask[size]=1
            if(mask[size]==1 and accumulator == 0):
                mask[size]=0  

        # use the mask to segment the TUGs (normalized)
        segm = zscore_normalization(i)*mask

        #print(mask)
        maskM.append(mask)
        segmT.append(segm)

        #### segmentacao eh ate aqui

    if savefile:
        np.save(filename, maskM)
    
    return maskM, segmT 


def read_segmentation_npy(filename="segmentation.pkl"):
    return np.load(filename)

 
## pdfs from data (x,y,z) / data_fusion


def generate_pdf_data(data):

    """Function to generate pdf from data (axes: x, y, z)"""

    N = max(data[0])
    j = 1
    
    for i in range(N):

        st = (i*3)
        datai = data.loc[st:st+2]
        x = np.asarray(datai[2][st])
        y = np.asarray(datai[2][st+1])
        z = np.asarray(datai[2][st+2])
        plt.plot(x)
        plt.plot(y)
        plt.plot(z)
        nome = "pdf_axes%03d.pdf"%j
        pp = PdfPages(nome)
        pp.savefig()
        pp.close()
        plt.clf()
        j = j + 1


def generate_pdf_fusion(fusion):
    
    """Function to generate Pdf from data fusion"""

    j = 1

    for i in fusion:
        plt.plot(i)
        nome = "pdf_fusion%03d.pdf"%j
        pp = PdfPages(nome)
        pp.savefig()
        pp.close()
        plt.clf()
        j = j + 1




def count_TUGs(series):
    
    ''' Counts the number of connected sequences of -1s inside
        the mask segmenting TUGs
        Returns:
            count: the number of TUGs detected by segmentation
            TUGsizes: a vector of 'count' elements, each with the
                    number of observations for each segmented TUG
            TUGpos: a vector of 'count' elements, each with the
                    starting position of each segmented TUG
    '''
    mask = series[0]
    segmentation = series[1]  
    
    for i in mask:
         
        count = 0
        curr = 0
        mask = series
    
        TUGpos = []

        # encontra primeiro '1'
        one = np.argmax(i[curr:])
        TUGpos.append(one) # guarda a primeira posicao

        curr = curr + one
        j = 1
        
        TUGsizes = []

        while (one > 0):
            #encontra o proximo '0'
            zero = np.argmin(i[curr:])
            print("TUG_%03d:"%j + str(zero))
            TUGsizes.append(zero)
            curr = curr + zero
            count = count + 1
        
            # encontra o proximo '1'
            one  = np.argmax(i[curr:])
            curr = curr + one
            TUGpos.append(curr)

            j = j + 1

        del TUGpos[-1] # remove last position

        return count, TUGsizes, TUGpos





        





