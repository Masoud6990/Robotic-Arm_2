import scipy.io as sio
import numpy
from numpy import savetxt 
from sklearn.decomposition import PCA

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

combinedData = numpy.empty([0, 3364])
labelOne = numpy.ones((22, 1))

for i in range (22):
	myKeys = loadmat("/content/drive/MyDrive/Colab Notebooks/Swati_files_single out put/files/Yes/eegreYes" + str(i+1) + ".mat")
	print(myKeys)
	eegData_re = myKeys['eeg_handle_re']
	eegData_orig = eegData_re.reshape(64, 512)
	eegData_orig = eegData_orig.transpose()
	eegData_1 = numpy.delete(eegData_orig, numpy.s_[57:61], axis=1)
	eegData_2 = numpy.delete(eegData_1, numpy.s_[14], axis=1)
	eegData = numpy.delete(eegData_2, numpy.s_[18], axis=1)
	print(eegData.shape)
	my_pca = PCA(n_components=58, random_state=2)
	my_pca.fit(eegData)
	print(my_pca.components_.shape)
	combinedData = numpy.append(combinedData, my_pca.components_.flatten().reshape(1, 3364), axis=0)

labelZero = numpy.zeros((22, 1))

for i in range (22):
	myKeys = loadmat("/content/drive/MyDrive/Colab Notebooks/Swati_files_single out put/files/No/eegreNo" + str(i+1) + ".mat")
	print(myKeys)
	eegData_re = myKeys['eeg_handle_re']
	eegData_orig = eegData_re.reshape(64, 512)
	eegData_orig = eegData_orig.transpose()
	eegData_1 = numpy.delete(eegData_orig, numpy.s_[57:61], axis=1)
	eegData_2 = numpy.delete(eegData_1, numpy.s_[14], axis=1)
	eegData = numpy.delete(eegData_2, numpy.s_[18], axis=1)
	print(eegData.shape)
	my_pca = PCA(n_components=58, random_state=2)
	my_pca.fit(eegData)
	print(my_pca.components_.shape)
	combinedData = numpy.append(combinedData, my_pca.components_.flatten().reshape(1, 3364), axis=0)


labels = numpy.append(labelOne, labelZero, axis=0)

combinedData = numpy.append(combinedData, labels, axis=1)
savetxt('/content/drive/MyDrive/Colab Notebooks/Swati_files_single out put/saved/combined_test_eeg.csv', combinedData, delimiter=',')

