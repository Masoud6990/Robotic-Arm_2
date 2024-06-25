import scipy.io as sio
import numpy
from numpy import savetxt 
from sklearn.decomposition import PCA
from tensorflow.random import set_seed
from numpy.random import seed
from sklearn.preprocessing import RobustScaler
from os import listdir
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter,filtfilt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

def _check_keys( dict):
	"""
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	"""
	for key in dict:
    		if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
        		dict[key] = _todict(dict[key])
	return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
	"""
	this function should be called instead of direct scipy.io .loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	"""
	data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	return _check_keys(data)

# setting the seed
seed(1)
set_seed(1)

# create the robust scaler for the data
rScaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(20, 100-20), unit_variance=True)

labels = []
epochs = []
my_input_data_List = []
input_data = []
final_input_data = []
input_to_nn = []
filtered_input_data = []
featuresList = []
totalLabelsList = []
num_rows = 500
num_cols = 64
epoch_size = 1024
channel_size = 64
window_size = 1000
decimated_size = 500
decimate_by = 2
polynomial_degree = 2
cols = 64
#[15, 17, 41, 43, 44, 47, 48, 49]

for mydir in [d for d in listdir("realtime_data")]:	
	print(mydir)
	
	for subdir in [s for s in listdir("realtime_data\\" + mydir)]:	
		print(subdir)

		for file in [f for f in listdir("realtime_data\\" + mydir + "\\" + subdir)]:
		
			try:
				myKeys = loadmat("realtime_data\\" + mydir + "\\" + subdir + "\\" + file)
				print("realtime_data\\" + mydir + "\\" + subdir + "\\" + file)
				#print(myKeys)
			except:
				continue

			eegData = myKeys['eeg_re'][0:epoch_size,0:channel_size]
			#print(eegData.shape)

			for k in range (0, window_size, decimate_by):
				epochs = eegData[k]
				my_input_data_List.append(epochs.reshape(1, 64))
			my_input_data_reshaped = numpy.reshape(my_input_data_List, (decimated_size, 64)) #500 x 8 			
			input_data = rScaler.fit_transform(my_input_data_reshaped)	
			input_data = input_data.transpose()
			input_to_nn = input_data.flatten().reshape(1, num_rows * num_cols)
			featuresList.append(input_to_nn)
			my_input_data_List = []
			filtered_input_data = numpy.empty((decimated_size, 0))

			if(subdir == 'Nomovement'):
				labels.append(0)
				print("0")
			if(subdir == 'Movement'):
				labels.append(1)
				print("1")

		final_input_data = numpy.array(featuresList)
		print("final_input_data shape")
		print(final_input_data.shape)
		
	#print(totalLabelsList)
myLabels = numpy.asarray(labels)
final_input_data_with_labels = numpy.append(final_input_data.reshape(len(final_input_data), (num_cols * num_rows)), numpy.array(labels).reshape(len(labels), 1), axis=1)
polyFeaturesList = []
final_input_data = numpy.empty((0, num_rows * num_cols))

savetxt('combined_RT_Training_Data.csv', final_input_data_with_labels, delimiter=',')

savetxt('combined_RT_Training_Data_Tr.csv', final_input_data_with_labels.transpose(), delimiter=',')