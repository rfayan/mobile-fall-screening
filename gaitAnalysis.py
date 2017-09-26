
import csv
import numpy as np
import matplotlib.pyplot as plt

from numpy import genfromtxt

data =genfromtxt('C:\\Users\\patricia\\Desktop\\Dados Acelerômetro\\001_Accelerometer_20170719-161640829.csv', delimiter=',')
lst= [elem for elem in data]
N = len(lst)
mat=np.bmat(lst)
mat=np.reshape(mat,(N,4))

plt.plot(mat[:,1])
plt.plot(mat[:,2])
plt.plot(mat[:,3])
plt.show()

matsquare=np.square(mat[:,1])+np.square(mat[:,2])+np.square(mat[:,3])
matfusao=np.sqrt(matsquare)




