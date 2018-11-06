import numpy as np
import matplotlib.pyplot as plt
import os
import regex as re
import scipy
import pandas as pd

from scipy import stats
from scipy.signal import butter, lfilter
from numpy import genfromtxt
from matplotlib.backends.backend_pdf import PdfPages


# Moacir
#directory = '/home/maponti/Repos/mobile-fall-screening/data/'

#Patricia
directory = 'C:\\Users\\Patrícia Bet\\Desktop\\Dados Acelerômetro\\'

#lista de idades(grupos)
index_60 = [0, 6, 8, 9, 10, 11, 13, 15, 18, 20, 23, 24, 25, 27, 28, 31, 33, 34, 35, 36, 37, 41, 43, 45, 46, 47, 49, 50, 52, 54, 55, 60, 63, 65, 69, 70, 72, 74, 75, 76]
index_70 = [1, 3, 4, 7, 14, 17, 21, 22, 32, 38, 39, 40, 42, 53, 56, 57, 58, 61, 62, 64, 71, 77]
index_80 = [2, 12, 19, 29, 30, 44, 51, 59, 67, 73, 78]


# listas de caidores e excluidos
index_faller   = [2, 4, 6, 7, 9, 10, 15, 20, 27, 29, 34, 35, 40, 46, 55, 58, 59, 63, 70, 77] #20 caidores
index_faller3M = [9, 10, 35, 40, 59, 70, 77] #7 caidores
index_faller6M = [7, 15, 27, 34, 46, 58, 59, 63] #8 caidores, 1 recorrente
index_faller9M = [2, 10] #2 caidores, 1 recorrente
index_faller12M = [4, 6, 20, 29, 55, 59, 63] #7 caidores, #2 recorrente
index_excluded = [5, 16, 26, 48, 65, 66] #6 excluídos

# dicionario com indices e meses
# podemos usar meses, e depois se precisar convertemos tudo para '1' (caidor)
dict_label = {2:9, 10:9, 7:6, 15:6, 27:6, 34:6, 46:6, 58:6, 59:6, 63:6, 9:3, 10:3, 35:3, 40:3, 59:3, 70:3, 77:3}

# dicionario com o numero de quedas
dict_qtde  = {7:1, 9:1, 10:2, 15:1, 24:1, 27:1, 34:1, 35:1, 40:1, 45:1, 58:1, 59:2, 63:1, 70:1, 77:1}



def data_read_csv(directory, so="Windows", savefile=True, filename="data_accelerometer.pkl"):

    ''' function to read csv files
        Parameters: 
            directory - 
    
        Returns:
            data - features matrix with all data (axes x, y and z)
    
    ''' 

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

    ''' function to generate the axes fusion
        Parameters: 
            matrix - data matrix

        Returns:
            dataf - data matrix with axes fusion

    '''

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

    ''' function to generate signal characteristics
        Parameters: 
            data - features matrix with axes fusion

        Return:
            featMatrix - matrix with all the features extracted from the signal
            (pse, psp1, psp2, psp3, pspf1, pspf2, pspf3, wpsp, cpt)
                      
    '''

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
        #plt.plot(espPotNyq, color='k')
        #plt.show()

       
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


        if (debug):
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
        j = j + 1
        featMatrix.append([id_vol] + features + 6*[0])

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
    
    ''' function to segment the signal corresponded to each TUG
        Parameters:
            tug - matrix with axes fusion 
            sumFilterSize - filter size

        Returns:
            mask - matrix with masks for each tug
                1 - tug
                0 - interval
            segment - matrix with signal for each tug   

    '''

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

    ''' counts the number of connected sequences of -1s inside
        the mask segmenting TUGs
        Parameters:
            mask - matrix with the masks

        Returns:
            count -  the number of TUGs detected by segmentation
            TUGsizes - a vector of 'count' elements, each with the
                    number of observations for each segmented TUG
            TUGpos - a vector of 'count' elements, each with the
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
    
    ''' counts the number of connected sequences of -1s inside
        the mask segmenting TUGs
        Parameters:
           mask  - matrix with the masks 
        
        Returns:
            count - the number of TUGs detected by segmentation
    
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
        Parameters: 
            count - the number of TUGs detected by segmentation 
            mask  - matrix with the masks of each tug

        Returns:
            count - the number of TUGs detected by segmentation
            mask  - matrix with the masks of each tug 

    '''

    countTUG = count[0]  # number of TUGs
    TUGsizes = count[1]  # size of each TUG
    TUGpos   = count[2]  # starting position of each TUG

    while len(TUGsizes) > 9:
        # starts with the TUG with minimum size
        minT = np.argmin(TUGsizes) 

        if minT == 0:
            mask[TUGpos[minT]:TUGpos[minT+1]] = 1
        elif minT == len(TUGsizes)-1:
            mask[TUGpos[minT-1]:TUGpos[minT]] = 1
        else:
            # test which side to merge with
            diff = [TUGpos[minT]-TUGpos[minT-1], TUGpos[minT+1]-TUGpos[minT]]
            print(diff)
            if diff[0] < diff[1]:
                mask[TUGpos[minT-1]:TUGpos[minT]] = 1
            else:
                mask[TUGpos[minT]:TUGpos[minT+1]] = 1

        # recompute the counting
        count = count_TUGs(mask)
        countTUG = count[0]  # number of TUGs
        TUGsizes = count[1]  # size of each TUG
        TUGpos   = count[2]  # starting position of each TUG

        if (debug):
            print("Min TUG: %d" % (minT))
            print("Mask position: %d" % (TUGpos[minT]))
            print('TUGsizes: ' + str(TUGsizes))
            print('TUGpos: ' + str(TUGpos))
            plt.plot(mask)
            plt.show()

    return count, mask


def label_TUG(newcount, newmask):
    
    ''' function to generate a labeled 'mascara', with 1, 2, 3, 4, 5, 6, 7, 8, 9
    for the different TUGs
    Parameters:
        newcount - TUG count corrected (must have 9 segmented TUGs)
            with: number of TUGs, sizes and positions
        newmask  - TUG mask corrected (1 for TUG positions, 0 for non-TUG signal)
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
    
    return newmask


## pdfs from data (x,y,z) / data_fusion
def generate_pdf_data(data, so = "all"):

    ''' function to generate pdf from data (axes: x, y, z)
        Parameters:
            data - data matrix
            so   - all = generate pdf from all the accelerometry data
                   ID  = generate pdf from each individual 

    '''

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

    ''' function to generate Pdf from data fusion
        Parameters:
            fusion - data matrix with axes fusion
            so     - all: generate pdf from all the accelerometry data
                     ID: generate pdf from each individual 

    '''

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

def statistic_Groups(matrix, featId, group1, group2, group3):

    ''' function for the variables statistical analysis of the three 
    initial groups using anova
        Parameters:
            matrix - the matrix with rows representing subjects
             first position is the ID, last position is the label
            featId - the feature to be tested ( > 0 ) 
            group1 - indices of 60-69 years old elderly
            group2 - indices of 70-79 years old elderly
            group3 - indices of 80 years old or more elderly

        Returns:
            anova - result of the anova test
            kruskal wallis - resul of the kruskal wallis test
    '''

    labPos = 9

    m60 = []
    m70 = []
    m80 = []

    #rotula na matriz
    for i in group1:
        matrix[i][labPos] = 1

    for i in group2:    
        matrix[i][labPos] = 2

    for i in group3:
        matrix[i][labPos] = 3

    #percorre todas as linhas
    for v in matrix:
        #monta as matrizes dos grupos
        if v[9] == 1:
            m60 = m60 + [v[featId]]
        elif v[9] == 2:
            m70 = m70 + [v[featId]]
        elif v[9] == 3: 
            m80 = m80 + [v[featId]]

    print("Feature %d" % (featId))
    print("\tMedias, grupo 60-69: %02.4f, desvio: %.4f" %(np.mean(m60), np.std(m60)))
    print("\tMedias, grupo 70-79: %02.4f, desvio: %.4f" %(np.mean(m70), np.std(m70)))
    print("\tMedias, grupo 80+  : %02.4f, desvio: %.4f" %(np.mean(m80), np.std(m80)))
    anova = stats.f_oneway(m60, m70, m80)
    kruskal_wallis = scipy.stats.kruskal(m60, m70, m80)
    print("\tp-value anova: %.4f " % (anova.pvalue))
    print("\tp-value kruskal wallis: %.4f " % (kruskal_wallis.pvalue))

    return anova, kruskal_wallis



def statistic_Fall(matrix, featId, indPos, indExc):

    ''' function for the variables statistical analysis of the two 
    groups formed from the two months of future fall observation 
    using test t
        Parameters:
            matrix - the matrix with rows representing subjects
             first position is the ID, last position is the label
            featId - the feature to be tested ( > 0 )
            indPos - indices of fallers
            indExc - indices of subject to be excluded

        Returns:
            tTest - result of the t test

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
    mannWhitney = scipy.stats.mannwhitneyu(mPos, mNeg, use_continuity=True, alternative=None) 
    print("\tp-value t test: %.4f " % (tTest.pvalue))
    print("\tp-value mannWhitney: %.4f " % (mannWhitney.pvalue))
 

    return tTest, mannWhitney


def save_masks(masks, filename='tug_masks.npy'):
    ''' save masks 
        Parameters:
            filename - with .npy extension
            masks    - array with masks for the participants' TUGs
    '''
    np.save(filename, masks)

def load_masks(filename='tug_masks.npy'):
    ''' load masks from file
        Parameters:
            filename - with .npy extension
    '''
    return np.load(filename)


###################

def TUG_features(data, mask, TUGs, filtering=True, savefile=True, filename="featuresAcc-TUG.npy"):
    ''' extracts features from segmented TUGs 
        Parameters:
            data - matrix with the axes fusion
            mask - matrix with the masks of each tug   
            TUGs - number of the tug execution to extrated the features (1 to 9)

        Returns:
            featTUGs - features extrated from the acceleration signal 
            corresponding to each tug
    
    
    '''

    print("Extracting features from TUGs: "+str(TUGs))

    dataseg = []
    for i in range(data.shape[0]):
    #for i in range(2):

        xd = data[i]  # data i
        xm = mask[i] # label i

        xm_count = count_TUGs(xm)
        xm_count, new_xm = correct_mask(xm_count, xm)
        xl = label_TUG(xm_count, new_xm)

        #plt.plot(xd)
        #plt.plot(xl)
        #plt.show()
        
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
     #   np.save(featTUGs)

    return featTUGs  


def read__TUGfeatures_npy(filename="featuresAcc-TUG.npy"):
    return np.load(filename)

   

def load_all_data():
    ''' loads signals (after fusion) and computes frequency
        features for the whole signals
    '''

    fusion = read_data_npy("data_fusion.npy")
    feat = data_features(fusion, filtering=True, debug=False)
    return fusion, feat


def label_features(matrix, indPos=index_faller, indExc=index_excluded):
    
    ''' labels the features matrix
        Parameters:
            matrix - features matrix, each row is an example
            indPos - indices for the fallers (positive examples)
                     default: index_faller (all fallers)
            indExc - indices for the excluded examples
                     default: index_excluded
        Returns:
            features matrix with label 1 for Positive,
            -1 for Negative and 0 for Excluded
    '''
 
    labPos = 10 # index of the label in the feature vector
    #rotula 
    for i in indPos:
        matrix[i][labPos] = 1   

    for i in indExc:
        matrix[i][labPos] = 0   

    labels = []
    feats = []
    for v in matrix:
        # skip the excluded ones
        if not(v[labPos] == 0):
            labels.append(v[labPos]) # get label
            feats.append(v[1:labPos])# get feats

    feats = np.asarray(feats)
    labels = np.asarray(labels)

    return feats, labels


def cutoff_points(feature, labels, n_cutoff=100, verbose=True):

    '''
        Parameters:
            features - feature matrix 
            labels   - label matrix with 1 for positive (fallers) 
             and -1 for negative (non-fallers)

        Returns:
           opt_predict - matrix with the predict condiction, 1 fallers 
            or -1 non-fallers
           opt_cut - cutoff point
           opt_acc - accuracy
           opt_sen - sensitivity
           opt_spe - specificity

    '''

    maxv = np.max(feature)
    minv = np.min(feature)
    opt_cut = 0
    opt_acc = 0
    opt_sen = 0
    opt_spe = 0

    # for each cuttof point 'c'
    for c in np.linspace(minv, maxv, n_cutoff):
        neg = np.where(feature < c)
        pos = np.where(feature >= c)

        # create the vector for the predicted labels
        predict = np.ones(len(labels)).astype(int)
        predict[neg] = -1
        
        true_neg = np.where(labels == -1)
        true_pos = np.where(labels == 1)

        correct = np.where(labels==predict)

        ACC = len(correct[0])/float(len(labels))

        TP = len(np.where(predict[true_pos]==1)[0])
        FP = len(np.where(predict[true_neg]==1)[0])
        FN = len(np.where(predict[true_pos]==-1)[0])
        TN = len(np.where(predict[true_neg]==-1)[0])

        # TPR == sensitivity
        TPR = TP/len(true_pos[0])
        # TNR == specificity
        TNR = TN/len(true_neg[0])

        if (np.abs(TPR-TNR) < 0.05):
            opt_predict = predict
            opt_cut = c
            opt_acc = ACC
            opt_sen  = TPR
            opt_spe  = TNR
        elif (c == minv): 
            opt_predict = predict
    
        if (verbose):
            if (c == opt_cut):
                print("*************** Equal Error Rate **************************")
            print("predict" % (predict))
            print("threshold %.2f, Accuracy=%.2f" % (c, ACC))
            print("\tTP=%.2f, TN=%.2f, FP=%.2f, FN=%.2f" % (TP, TN, FP, FN))
            print("\tSensitivity (TPR)=%.2f, Specificity (TNR)=%.2f\n" % (TPR, TNR))

    print("Optimal threshold: %.3f, Acc=%.4f, Sens=%.4f, Spec=%.4f" % (opt_cut, opt_acc, opt_sen, opt_spe))

    return opt_predict



def late_fusion(feat, label):
    ''' Performs fusion on the decisions'''

    predict  = []
    decision = []

    for col in range(feat.shape[1]):
    #for col in [0,2,3,5,6]:
        p = cutoff_points(feat[:,col],label, verbose=False)
        predict.append(p)
    
    # optimal predictions for each feature
    predict = np.array(predict) 
    
    # combine using majority voting (median)
    for col in range(predict.shape[1]):
        d = np.median(predict[:,col])
        decision.append(d)
    
    decision = np.asarray(decision)
    decision.astype(int)
    
    correct = np.where(label == decision)
 
    true_neg = np.where(label == -1)
    true_pos = np.where(label == 1)

    ACC = len(correct[0])/float(len(label))
    TP = len(np.where(decision[true_pos]==1)[0])
    FP = len(np.where(decision[true_neg]==1)[0])
    FN = len(np.where(decision[true_pos]==-1)[0])
    TN = len(np.where(decision[true_neg]==-1)[0])

    # TPR == sensitivity
    TPR = TP/len(true_pos[0])
    # TNR == specificity
    TNR = TN/len(true_neg[0])

    print("Late fusion:")
    print("\tAccuracy=%.4f" % (ACC))
    print("\tTP=%.4f, TN=%.4f, FP=%.4f, FN=%.4f" % (TP, TN, FP, FN))
    print("\tSensitivity (TPR)=%.4f, Specificity (TNR)=%.4f\n" % (TPR, TNR))

    return predict, decision, correct


def early_fusion(feat, label):
    ''' Performs fusion on the features '''

    normalized_feat = []
    average_feat = []
    std_feat = []
    decision = []

    # for each column (feature), normalize 0-1
    nrows, nfeats = feat.shape
    for i in range(nfeats):
        f = feat[:,i]
        n = (f-min(f))/(max(f)-min(f))
        normalized_feat.append(n)

    normalized_feat = np.asarray(normalized_feat)

    # for each, compute mean/std of the normalized features
    for i in range(nrows):
        e = normalized_feat[:,i]
        m = np.mean(e)
        dp = np.std(e)
        average_feat.append(m)
        std_feat.append(dp)
    
    # invert feature (check if needed)
    average_feat = np.asarray(average_feat)
    std_feat = np.asarray(std_feat)

    d_average = cutoff_points(average_feat, label, verbose=False)
    d_std = cutoff_points(std_feat, label, verbose=False)
    decision.append(d_average)
    decision.append(d_std)
    decision = np.asarray(decision)

    return decision



def load_data():
    data = read_data_pkl()
    fusion = read_data_npy()
    mask= load_masks()
    segm = read_segmentation_npy()

    return data, fusion, mask, segm
