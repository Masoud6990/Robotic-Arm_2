import numpy
from tensorflow.random import set_seed
from numpy.random import seed
import os

# setting the seed
seed(1)
set_seed(1)

loaded_complete_data_rt = numpy.loadtxt('combined_RT_Training_Data.csv', delimiter=',')
loaded_complete_data_tr = numpy.loadtxt('combined_Training_Data.csv', delimiter=',')

loaded_complete_data = numpy.concatenate((loaded_complete_data_rt, loaded_complete_data_tr), axis=0)	

print(loaded_complete_data.shape)

numpy.savetxt('combined_data_2_classes.csv', loaded_complete_data, delimiter=',')	

transposed_data = numpy.transpose(loaded_complete_data)
numpy.savetxt('transposed_data.csv', transposed_data, delimiter=',')