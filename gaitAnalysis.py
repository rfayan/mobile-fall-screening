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
index_faller   = [7, 9, 10, 15, 24, 27, 34, 35, 40, 45, 58, 59, 63, 70, 77]
index_faller3M = [9, 10, 35, 40, 59, 70, 77]
index_faller6M = [7, 15, 27, 34, 46, 58, 59, 63]
index_faller9M = [10, 24]
index_excluded = [5, 16, 26, 48, 65, 66]

# dicionario com indices e meses
# podemos usar meses, e depois se precisar convertemos tudo para '1' (caidor)
dict_label = {10:9, 24:9, 7:6, 15:6, 27:6, 34:6, 46:6, 58:6, 59:6, 63:6, 9:3, 10:3, 35:3, 40:3, 59:3, 70:3, 77:3}

# dicionario com o numero de quedas
dict_qtde  = {7:1, 9:1, 10:2, 15:1, 24:1, 27:1, 34:1, 35:1, 40:1, 45:1, 58:1, 59:2, 63:1, 70:1, 77:1}



def data_read_csv(directory, so="Windows", savefile=True, filename="data_accelerometer.pkl"):

    '''Function to read csv files''' 

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

                dataf = genfromtxt(f, delimiter=',')
                lst = [elem for elem in dataf]
                N = len(lst)
                mat = np.bmat(lst)
                mat = np.reshape(mat,(N,4))


                jId = [j,'x'] # cria id com eixo x
                jId.append(np.squeeze(np.asarray(mat[1:,1]))) # adiciona dados x
                data.append(jId)
                jId = [j,'y'] # cria id com eixo y
                jId.append(np.squeeze(np.asarray(mat[1:,2]))) # adiciona dados y
                data.append(jId)
                jId = [j,'z'] # cria id com eixo z
                jId.append(np.squeeze(np.asarray(mat[1:,3]))) # adiciona dados z
                data.append(jId)

    data = pd.DataFrame(data)

    # saves the read data into a file               
    if savefile:
        data.to_pickle(filename)

    return data


def read_data_pkl(filename="data_accelerometer.pkl"):
    return pd.read_pickle(filename)

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S

    ''' Using numpy strides to apply function along arrays
        OBS: thanks to Divakar 
        https://stackoverflow.com/users/3293881/divakar
        '''
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n)) 

def median_filter(data, window_len=3, axis=1):
    return np.median(strided_app(data, window_len, 1), axis=axis)



##fusion from data_read_csv

def data_fusion(data, savefile=True, filename="data_fusion.npy"):

    '''Function to generate the axes fusion'''

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


def read_data_npy(filename="data_fusion.npy"):
    return np.load(filename)



## features from data_fusion

def data_features(data, filtering=True, debug=False, savefile=True, filename="featuresAcc.npy"):

    '''Function to generate signal characteristics'''

    featMatrix = []
    j = 1

    for i in data:
        # filtering
        if filtering == True:
            fs = 100
            datafilt = butter_lowpass_filter(i, 3.667, fs, order=3)
            dataF = np.fft.fft(datafilt)
        else:
            # Fourier Transform of the data 
            dataF = np.fft.fft(i)

        # Power Spectrum of the data
        espPot = np.power(np.abs(dataF),2)
 
        # number of frequencies to be displayed
        n = espPot.size
        nhalf = int(n/2)
        maxfr = 50
        timestep = 0.1
        freq = np.fft.fftfreq(n,timestep)

        espPotNyq = espPot[1:maxfr]

        #1-power spectral entropy
        #computed over the Spectrum normalized between 0-1
        espProb = espPotNyq.copy()
        espProb = (espProb - np.min(espProb)) / (np.max(espProb) - np.min(espProb))
        pse = -sum(espProb*np.log(0.0001 + espProb)) 

        # the remaining features are computed over a
        # z-score normalization (i.e. x-mean/sd)
        espPotNyq = (espPotNyq-np.mean(espPotNyq)) / np.std(espPotNyq)

        #2-power spectrum peak
        psp1 = np.max(espPotNyq)

        #3-power spectrum peak frequency
        pspf1 = np.argmax(espPotNyq)

        #4-weighted power spectrum peak
        wpsp = np.argmax(espPotNyq)*np.max(espPotNyq)

        # get second peak pspf / psp (removing 3 psp1 neighbours)
        espPotNyq[pspf1] = 0
        cut3 = max(0,pspf1-3)
        espPotNyq[cut3:pspf1] = 0
        cut3 = min(len(espPotNyq),pspf1+3)
        espPotNyq[pspf1:(cut3+1)] = 0

        pspf2 = np.argmax(espPotNyq)
        psp2  = np.max(espPotNyq)

        # get third peak pspf / psp (removing 2 psp2 neighbours)
        espPotNyq[pspf2] = 0
        cut3 = max(0,pspf2-2)
        espPotNyq[cut3:pspf2] = 0
        cut3 = min(len(espPotNyq),pspf2+2)
        espPotNyq[pspf2:(cut3+1)] = 0
        
        pspf3 = np.argmax(espPotNyq)
        psp3  = np.max(espPotNyq)

        #5- cpt
        # generate weights for 1 to 50 Hz
        # weights = (50-np.arange(50))/50.0 # weights from 1 to ~0
        weights = (50-np.arange(0,50,0.5))/50.0
        weights = weights[:maxfr]

        # weight frequency coefficients 
        maxF = np.max(espPot[:maxfr])
        freqw = espPot[:maxfr]*weights
        maxfp = np.max(freqw)

        cpt = (np.sum(freqw/maxfp)-1)*100.0

        pspf1 = pspf1+1
        pspf2 = pspf2+1
        pspf3 = pspf3+1
        print("Features:")
        print("PSE = " + str(pse))
        print("PSP = " + str(psp1) + " " + str(psp2) + " " + str(psp3))
        print("PSPF = " + str(pspf1) + " " + str(pspf2) + " " + str(pspf3))
        print("WPSP = " + str(wpsp))
        print("CPTs = " + str(cpt))

        if (debug):
            plt.plot(np.arange(1,maxfr), espPot[1:maxfr], color='k')
            plt.axvline(x=int(pspf1),color='r', linestyle='-') 
            plt.axvline(x=int(pspf2),color='g', linestyle='--') 
            plt.axvline(x=int(pspf3),color='b', linestyle=':') 
            plt.show()

        features = [pse, psp1, psp2, psp3, pspf1, pspf2, pspf3, wpsp, cpt]
        id_vol = "%03d"%j
        months = -1
        j = j + 1
        featMatrix.append([id_vol] + features + [months])

    if savefile:
        np.save(filename, featMatrix)

    return featMatrix


def read_featuresAcc_npy(filename="featuresAcc.npy"):
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


def zscore_normalization(signal):
    return (signal-np.mean(signal))/np.std(signal)


## segmentation from data_fusion

def data_segmentTS_TUG(tug, sumFilterSize=300, savefile=True, filename="segmentation.npy"):

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


        mask = median_filter(mask, window_len=50, axis=1)
        #print(mask)
        maskM.append(mask)
        segmT.append(segm)

    if savefile:
        np.save(filename, maskM)

    return maskM, segmT 


def read_segmentation_npy(filename="segmentation.npy"):
    return np.load(filename)




def count_TUGs(mask):

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
    TUGpos = []

    # encontra primeiro '1'
    one = np.argmax(mask[curr:])
    TUGpos.append(one) # guarda a primeira posicao

    curr = curr + one
    j = 1

    TUGsizes = []

    while (one > 0):
        #encontra o proximo '0'
        zero = np.argmin(mask[curr:])
        print("TUG_%03d:"%j + str(zero))
        TUGsizes.append(zero)
        curr = curr + zero
        count = count + 1

        # encontra o proximo '1'
        one  = np.argmax(mask[curr:])
        curr = curr + one
        TUGpos.append(curr)

        j = j + 1

    del TUGpos[-1] # remove last position

    return count, TUGsizes, TUGpos


def count_TUGs_all(mask):
    
    ''' Counts the number of connected sequences of -1s inside
        the mask segmenting TUGs
        Returns:
            count: the number of TUGs detected by segmentation
            TUGsizes: a vector of 'count' elements, each with the
                    number of observations for each segmented TUG
            TUGpos: a vector of 'count' elements, each with the
                    starting position of each segmented TUG
    '''
    newcount = []

    for i in mask:
        count = count_TUGs(mask[i])
        
        newcount.append(count)
        
        #print(m)
    
    return newcount

def correct_mask(count, mask, debug=False): 

    ''' checks the size of each TUG (observations)
        in order to split those above some max threshold (max_size)
        and merge those below some minimum threshold (min_size)
    '''

    countTUG = count[0]  # number of TUGs
    TUGsizes = count[1]  # size of each TUG
    TUGpos   = count[2]  # starting position of each TUG

    while len(TUGsizes) > 9:
        # starts with the TUG with minimum size
        minT = np.argmin(TUGsizes) 

        # get positions of minT and its neighbours
        currT = TUGpos[minT]
        prevT = TUGpos[minT-1] + TUGsizes[minT-1]
        nextT = TUGpos[minT+1]

        # test which side to merge with
        # 0 - previous, 1 - next
        diff = [currT-prevT, nextT-currT]
        print(diff)
        if np.argmin(diff) == 0:
            mask[prevT-1:currT+1] = 1
        else:
            mask[currT:nextT+1] = 1

        # recompute the counting
        count = count_TUGs(mask)
        countTUG = count[0]  # number of TUGs
        TUGsizes = count[1]  # size of each TUG
        TUGpos   = count[2]  # starting position of each TUG

        if (debug):
            print("Min TUG: %d" % (minT))
            print("Mask position: %d" % (currT))
            print(TUGsizes)
            print(TUGpos)
            plt.plot(mask)
            plt.show()

    return count, mask


def label_TUG(newcount, newmask):
    
    ''' Function to generate a labeled 'mascara', with 1, 2, 3, 4, 5, 6, 7, 8, 9
    for the different TUGs
    Parameters:
        newcount = TUG count corrected (must have 9 segmented TUGs) with: number of TUGs, sizes and positions
        newmask  = TUG mask corrected (1 for TUG positions, 0 for non-TUG signal)
    '''
    
    # splits newcount data into TUG count, sizes and positions
    countTUG= newcount[0]
    TUGsizes = newcount[1]
    TUGpos = newcount[2]

    ID = 0
    for j in range(countTUG):
        tug = TUGpos[ID] + TUGsizes[ID]
        newmask[TUGpos[ID]:tug] = ID + 1
        ID = ID + 1
    
    '''        
    tug1 = TUGpos[0] + TUGsizes[0]
    newmask[TUGpos[0]:tug1] = 1

    tug2 = TUGpos[1] + TUGsizes[1]
    newmask[TUGpos[1]:tug2] = 2

    tug3 = TUGpos[2] + TUGsizes[2]
    newmask[TUGpos[2]:tug3] = 3

    tug4 = TUGpos[3] + TUGsizes[3] 
    newmask[TUGpos[3]:tug4] = 4

    tug5 = TUGpos[4] + TUGsizes[4] 
    newmask[TUGpos[4]:tug5] = 5

    tug6 = TUGpos[5] + TUGsizes[5]
    newmask[TUGpos[5]:tug6] = 6

    tug7 = TUGpos[6] + TUGsizes[6]
    newmask[TUGpos[6]:tug7] = 7

    tug8 = TUGpos[7] + TUGsizes[7] 
    newmask[TUGpos[7]:tug8] = 8

    tug9 = TUGpos[8] + TUGsizes[8]
    newmask[TUGpos[8]:tug9] = 9

    '''
    
    return newmask


## pdfs from data (x,y,z) / data_fusion
def generate_pdf_data(data, so = "all"):

    '''Function to generate pdf from data (axes: x, y, z)'''

    if so == "all":
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


    elif so == "ID":
        IDi = int(input('Número de identificação do voluntário na matriz:'))
        ID = IDi - 1
        st = ID * 3
        datai = data.loc[st:st+2]
        x = np.asarray(datai[2][st])
        y = np.asarray(datai[2][st+1])
        z = np.asarray(datai[2][st+2])
        plt.plot(x)
        plt.plot(y)
        plt.plot(z)
        #plt.show()
        nome = "pdf_axesID%03d.pdf"%IDi
        pp = PdfPages(nome)
        pp.savefig()
        pp.close()
        plt.clf()



def generate_pdf_fusion(fusion, so="all"):

    '''Function to generate Pdf from data fusion'''

    if so == "all":

        j = 1
        for i in fusion:
            plt.plot(i)
            nome = "pdf_fusion%03d.pdf"%j
            pp = PdfPages(nome)
            pp.savefig()
            pp.close()
            plt.clf()
            j = j + 1

    elif so == "ID":
        IDi = int(input('Número de identificação do voluntário na matriz:'))
        ID = IDi - 1
        IDmatrix = fusion[ID]
        plt.plot(IDmatrix)
        #plt.show()
        nome = "pdf_fusionID%03d.pdf"%IDi
        pp = PdfPages(nome)
        pp.savefig()
        plt.clf()



#analysis

def anovaGroups(matrix, featId, group1, group2, group3):

    '''Function for the variables statistical analysis of the three 
    initial groups using anova'''

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


def tTestFeatures(matrix, featId, indPos, indExc):

    '''Function for the variables statistical analysis of the two 
    groups formed from the two months of future fall observation 
    using test t

    matrix - the matrix with rows representing subjects
             first position is the ID, last position is the label
    featId - the feature to be tested ( > 0 )

    indPos - indices of fallers

    indExc - indices of subject to be excluded

    '''

    labPos = 10 # index of the label in the feature vector

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


def save_masks(masks, filename='tug_masks.npy'):
    '''save masks 
        Parameters:
            filename - with .npy extension
            masks    - array with masks for the participants' TUGs
    '''
    np.save(filename, masks)

def load_masks(filename='tug_masks.npy'):
    '''load masks from file
        Parameters:
            filename - with .npy extension
    '''
    return np.load(filename)


###################
def runExample():
    fusion = read_data_npy("data_fusion.npy") 
    mask, segm = data_segmentTS_TUG(fusion)
    count5 = count_TUGs(mask[5])
    correct5 = correct_mask(count5, mask[5])
    label = label_TUG(correct5[0], correct5[1])


def TUG_features(data, mask, TUGs,  filtering=True, savefile=True, filename="featuresAcc-TUG.npy"):
    ''' Extracts features from segmented TUGs  '''

    print("Extracting features from TUGs: "+str(TUGs))

    dataseg = []
    for i in range(data.shape[0]):
    #for i in range(2):

        xd = data[i]  # data i
        xm = mask[i] # label i

        xm_count = count_TUGs(xm)
        xm_count, new_xm = correct_mask(xm_count, xm)
        xl = label_TUG(xm_count, new_xm)

        plt.plot(xd)
        plt.plot(xl)
        plt.show()
        
        # creates a new signal from xd, containing only the 
        # segmented labels at xl, defined by TUGs
        newx = []

        for l in TUGs:
            newx = np.concatenate((newx, xd[np.where(xl == l)]))

        #plt.plot(newx)
        #plt.show()

        dataseg.append(newx)

    featTUGs = data_features(dataseg, filtering=filtering, savefile=savefile, filename=filename)

    #if savefile:
     #   np.save(featTUGs, dataseg)

    return featTUGs #dataseg 


def read__TUGfeatures_npy(filename="featuresAcc-TUG.npy"):
    return np.load(filename)



def correct_manualTUG(mask):

    mask[0][8350:8528] = 1

    mask[2][1954:2976] = 1
    mask[2][6115:6364] = 1 #

    mask[3][:369] = 0
    mask[3][391:901] = 1
    mask[3][2740:3638] = 1
    mask[3][3707:5139] = 0

    mask[4][2968:4007] = 1
    mask[4][11794:12944] = 1

    mask[6][7718:7808] = 1
    mask[6][8978:9113] = 1

    mask[7][2499:2948] = 0

    mask[8][2513:2993] = 1

    mask[10][8256:8689] = 1

    mask[12][8514:8767] = 1
    
    mask[13][2727:3314] = 0
    mask[13][3337:4362] = 1
    mask[13][10366:10563] = 0
    mask[13][10599:11696] = 1

    mask[14][1564:2305] = 1
    mask[14][3170:4198] = 1
    mask[14][6810:6926] = 0
    mask[14][7512:7736] = 1
    mask[14][10739:11134] = 1

    mask[15][6440:7803] = 1

    mask[16][560:570] = 0
    mask[16][842:1152] = 0
    mask[16][1508:1526] = 0
    mask[16][1927:2301] = 0
    mask[16][2566:2848] = 0
    mask[16][3040:3067] = 0
    mask[16][3459:3541] = 0
    mask[16][3815:3869] = 0

    mask[17][2728:2910] = 1
    mask[17][4241:4402] = 1
    mask[17][5369:5583] = 1
    mask[17][6485:6721] = 1
    mask[17][9566:9749] = 1
    
    mask[22][7849:7967] = 1
    mask[22][9771:9964] = 1

    mask[24][130:750] = 1
    mask[24][5129:6044] = 1
    mask[24][8498:9169] = 1

    mask[25][3714:3934] = 1
    mask[25][6273:6472] = 1

    mask[27][5157:6064] = 1

    mask[28][3331:4069] = 1
    mask[28][1666:2163] = 0

    mask[30][11130:11360] = 1
    mask[30][11384:11553] = 0

    mask[32][1031:1344] = 1
    mask[32][8279:8407] = 1

    mask[33][2714:3663] = 1
    mask[33][8113:8283] = 1

    mask[35][3792:4913] = 1

    mask[36][921:1122] = 1
    mask[36][3532:3754] = 1
    mask[36][6609:6757] = 1
    mask[36][9622:9897] = 1

    mask[38][2747:3245] = 0
    mask[38][3290:4198] = 1

    mask[40][1280:1557] = 1
    mask[40][2432:2594] = 1
    mask[40][3769:4080] = 1
    mask[40][7122:7456] = 1
    mask[40][9669:10671] = 1

    mask[42][10202:10802] = 0

    mask[44][730:1531] = 1
    mask[44][2003:2717] = 1
    mask[44][3947:5605] = 1
    mask[44][8922:10366] = 1
    mask[44][13210:13696] = 1

    mask[49][987:1735] = 1
    mask[49][2322:2858] = 1
    mask[49][3981:4954] = 1
    mask[49][10719:11630] = 1

    mask[50][1791:2279] = 1
    mask[50][2317:2825] = 0
    mask[50][5314:5476] = 0
    mask[50][5496:6367] = 1
    mask[50][6874:7066] = 1
    mask[50][8626:8808] = 1

    mask[54][:1145] = 0
    mask[54][1146:1806] = 1
    mask[54][1809:2513] = 0
    mask[54][2511:3105] = 1

    mask[55][1357:1924] = 1
    mask[55][2268:2936] = 1
    mask[55][5807:5965] = 1

    mask[56][5439:5765] = 0
    mask[56][5767:6390] = 1
    mask[56][10440:10578] = 0
    mask[56][10579:11803] = 1

    mask[58][4511:4850] = 1
    mask[58][10553:10943] = 0

    mask[59][:1417] = 0
    mask[59][3470:4928] = 1
    mask[59][8257:9492] = 1
    mask[59][10921:11754] = 1
    mask[59][11823:12045] = 0
    mask[59][12073:13100] = 1

    mask[61][2418:3072] = 1
    mask[61][5610:6124] = 1

    mask[62][7885:8287] = 1
    mask[62][11248:11504] = 1

    mask[64][5928:6062] = 1

    mask[65][2103:2747] = 1
    mask[65][5356:6318] = 1
    mask[65][9402:9772] = 1

    mask[69][1337:1813] = 1
    mask[69][2621:2911] = 1
    mask[69][5967:6288] = 1
    mask[69][9509:9716] = 1

    mask[70][3115:3550] = 1
    mask[70][6220:6445] = 1

    mask[71][2582:2809] = 1
    mask[71][5845:6138] = 1
    mask[71][8606:8852] = 1

    mask[72][2456:3172] = 1
    mask[72][5812:6296] = 1
    mask[72][7970:8648] = 1

    mask[74][3406:3615] = 1
    mask[74][6195:6442] = 1
    mask[74][8709:8842] = 1

    mask[75][2778:2945] = 1
    mask[75][6184:6596] = 1
    mask[75][8834:9168] = 1
    
    mask[77][2664:3647] = 1

    mask[78][8594:8925] = 1
    mask[78][10718:11422] = 1
    mask[78][12111:13063] = 1

    return mask 


