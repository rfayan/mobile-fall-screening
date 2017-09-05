
>>> import csv
>>> import numpy as np
>>> from numpy import genfromtxt
>>> 
>>> data =genfromtxt('C:\\Users\\patricia\\Desktop\\Dados Acelerômetro\\001_Accelerometer_20170719-161640829.csv', delimiter=',')
>>> lst= [elem for elem in data]
>>> N = len(lst)
>>> mat=np.bmat(lst)
>>> mat=np.reshape(mat,(N,4))
>>> 
>>> 
>>> import matplotlib.pyplot as plt
>>> plt.plot(mat[:,1])
[<matplotlib.lines.Line2D object at 0x000001504761D588>]
>>> plt.plot(mat[:,2])
[<matplotlib.lines.Line2D object at 0x000001504762F4E0>]
>>> plt.plot(mat[:,3])
[<matplotlib.lines.Line2D object at 0x000001504762FD68>]
>>> plt.show()
>>> 
>>> 
>>> matsquare=np.square(mat[:,1])+np.square(mat[:,2])+np.square(mat[:,3])
>>> matfusao=np.sqrt(matsquare)


