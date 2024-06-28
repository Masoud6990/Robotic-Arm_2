from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import tensorflow
from numpy import mean
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import minmax_scale
from numpy.random import seed
from tensorflow.random import set_seed

# setting the seed
seed(1)
set_seed(1)

rScaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(20, 100-20), unit_variance=True)

# load the test data
X = loadtxt('/content/drive/MyDrive/Colab Notebooks/Swati_files_single out put/saved/single_test_input.csv', delimiter=',')

my_input = rScaler.fit_transform(X[0:1740].reshape(1, 1740).transpose())
single_input = my_input.transpose()

# transform the data in the format which the model wants
single_input = single_input.reshape(1, 30, 58)

# load the model
model = load_model('/content/drive/MyDrive/Colab Notebooks/Swati_files_single out put/saved/best_model.h5')

# get the "predicted class" outcome
y_hat = model.predict(single_input) 
y_pred = numpy.argmax(y_hat,axis=-1)
print(y_pred)
