import csv
import numpy as np
import matplotlib.pyplot as plt
import os

from numpy import genfromtxt
from matplotlib.backends.backend_pdf import PdfPages


def generatePdfFromCsv(directory = 'C:\\Users\\patricia\\Desktop\\Dados Acelerômetro\\'):
        j=1

        for root, dirs, files in os.walk(directory):
                for f in files:
                        if f.endswith(".csv"):
                                f=directory+f
                                data =genfromtxt(f, delimiter=',')
                                lst= [elem for elem in data]
                                N = len(lst)
                                mat = np.bmat(lst)
                                mat = np.reshape(mat,(N,4))
                                plt.plot(mat[:,1])
                                plt.plot(mat[:,2])
                                plt.plot(mat[:,3])
                                nome="dados%03d.pdf"%j
                                pp = PdfPages(nome)
                                pp.savefig()
                                pp.close()
                                plt.clf()
                                j=j+1


def plot(x):
        plt.plot(x[2])
        plt.title(x[1])
        plt.show()

 
def createMatrixFromCsv(directory = 'C:\\Users\\patricia\\Desktop\\Dados Acelerômetro\\'):
        j=1
        dataAcc = [] # matriz vazia
        for root, dirs, files in os.walk(directory):
                for f in files:
                        if f.endswith(".csv"):
                                f=directory+f
                                data =genfromtxt(f, delimiter=',')
                                lst= [elem for elem in data]
                                N = len(lst)
                                mat = np.bmat(lst)
                                mat = np.reshape(mat,(N,4))
                                jId=[j,'x'] # cria id com eixo x
                                jId.append(mat[:,1]) # adiciona dados
                                dataAcc.append(jId)
                                jId=[j,'y'] # cria id com eixo x
                                jId.append(mat[:,2]) # adiciona dados
                                dataAcc.append(jId)
                                jId=[j,'z'] # cria id com eixo x
                                jId.append(mat[:,3]) # adiciona dados
                                dataAcc.append(jId)
        return dataAcc                          
                                


generatePdfFromCsv()
matrizDados = createMatrixFromCsv()
