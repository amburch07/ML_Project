# dataset:
# https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

# x, y, and z accelerometer data (linear acceleration) and gyroscopic data (angular velocity) from the smart phone, specifically a Samsung Galaxy S II. Observations were recorded at 50 Hz

# Pre-processing accelerometer and gyroscope using noise filters.
# Splitting data into fixed windows of 2.56 seconds (128 data points) with 50% overlap.
# Splitting of accelerometer data into gravitational (total) and body motion components.

# The dataset was split into train (70%) and test (30%) sets based on data for subjects, e.g. 21 subjects for train and nine for test.

# raw data: total acceleration, body acceleration, and body gyroscope. Each has three axes of data. This means that there are a total of nine variables for each time step.

# Further, each series of data has been partitioned into overlapping windows of 2.65 seconds of data, or 128 time steps. These windows of data correspond to the windows of engineered features (rows) in the previous section.

# This means that one row of data has (128 * 9), or 1,152, elements. 

#read to numpy 3d:     [samples, 128 time steps, 9 features]
# trainX.shape (7352, 128, 9)
# testX shape: (2947, 128, 9)

# cnn model
import tensorflow as tf
from numpy import mean
from numpy import std
from numpy import stack, vstack, hstack, dstack
from numpy import pad
from numpy import newaxis
from sys import platform

from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers import BatchNormalization, Activation, GlobalAveragePooling1D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras.utils import to_categorical
#welch
from scipy import signal

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    #add more feature
    # trainX, testX = add_freq(trainX, testX)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy

#stack psd of 3-axes together as another dimension.
#add 2 more depth: from 9 to 11
def add_freq(trainX, testX):
    #add trainX total accel and body g
    # # f, Pxx = signal.welch(trainX[:,:,0], fs=50, nperseg=128)
    # Pxx.shape: (7352, 65)
    f, Pxx_total_acc_x = signal.welch(trainX[:,:,0], fs=50, nperseg=64)
    f, Pxx_total_acc_y = signal.welch(trainX[:,:,1], fs=50, nperseg=64)
    f, Pxx_total_acc_z = signal.welch(trainX[:,:,2], fs=50, nperseg=64)
    Pxx_total_acc_xyz = hstack((Pxx_total_acc_x,Pxx_total_acc_y, Pxx_total_acc_z))
    # Pxx_total_acc_xyz 7352,99
    Pxx_total_acc_xyz = pad(Pxx_total_acc_xyz, 15, pad_with)[15:-15,:128]
    # (7352, 128)
    Pxx_total_acc_xyz = Pxx_total_acc_xyz[..., newaxis]
    # (7352, 128,1)
    f, Pxx_body_gyro_x = signal.welch(trainX[:,:,6], fs=50, nperseg=64)
    f, Pxx_body_gyro_y = signal.welch(trainX[:,:,7], fs=50, nperseg=64)
    f, Pxx_body_gyro_z = signal.welch(trainX[:,:,8], fs=50, nperseg=64)
    Pxx_body_gyro_xyz = hstack((Pxx_body_gyro_x,Pxx_body_gyro_y, Pxx_body_gyro_z))
    Pxx_body_gyro_xyz = pad(Pxx_body_gyro_xyz, 15, pad_with)[15:-15,:128]
    Pxx_body_gyro_xyz = Pxx_body_gyro_xyz[..., newaxis]
    #add to train
    trainX = dstack((trainX,Pxx_total_acc_xyz,Pxx_body_gyro_xyz))

    # repeat for testX
    f, Pxx_total_acc_x = signal.welch(testX[:,:,0], fs=50, nperseg=64)
    f, Pxx_total_acc_y = signal.welch(testX[:,:,1], fs=50, nperseg=64)
    f, Pxx_total_acc_z = signal.welch(testX[:,:,2], fs=50, nperseg=64)
    Pxx_total_acc_xyz = hstack((Pxx_total_acc_x,Pxx_total_acc_y, Pxx_total_acc_z))
    Pxx_total_acc_xyz = pad(Pxx_total_acc_xyz, 15, pad_with)[15:-15,:128]
    Pxx_total_acc_xyz = Pxx_total_acc_xyz[..., newaxis]
    # (7352, 128,1)
    f, Pxx_body_gyro_x = signal.welch(testX[:,:,6], fs=50, nperseg=64)
    f, Pxx_body_gyro_y = signal.welch(testX[:,:,7], fs=50, nperseg=64)
    f, Pxx_body_gyro_z = signal.welch(testX[:,:,8], fs=50, nperseg=64)
    Pxx_body_gyro_xyz = hstack((Pxx_body_gyro_x,Pxx_body_gyro_y, Pxx_body_gyro_z))
    Pxx_body_gyro_xyz = pad(Pxx_body_gyro_xyz, 15, pad_with)[15:-15,:128]
    Pxx_body_gyro_xyz = Pxx_body_gyro_xyz[..., newaxis]
    #add to train
    testX = dstack((testX,Pxx_total_acc_xyz,Pxx_body_gyro_xyz))
    return trainX, testX

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]-1] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, use_bias=False, input_shape=(n_timesteps,n_features)))
    model.add(BatchNormalization(scale=False))
    model.add(Activation("relu"))

    model.add(Conv1D(filters=64, kernel_size=3, use_bias=False))
    model.add(BatchNormalization(scale=False))
    model.add(Activation("relu"))

    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))

# #block 2
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(Dropout(0.25))
    model.add(GlobalAveragePooling1D())
# #block 3
# model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
# model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
# model.add(Dropout(0.25))
# model.add(MaxPooling1D(pool_size=2))

# #block 4
# model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
# model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
# model.add(Dropout(0.25))
# model.add(MaxPooling1D(pool_size=2))

#return
    # model.add(Flatten())
    model.add(Dense(100, use_bias=False))
    model.add(BatchNormalization(scale=False))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Dense(300, use_bias=False))
    model.add(BatchNormalization(scale=False))
    model.add(Activation("relu"))
    model.add(Dense(n_outputs, activation='softmax'))

    #disp size
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    #ME: save model
    model.save('HARModel1D_batchNorm.h5')
    return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=3):
    # load data
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


# run the experiment
run_experiment(1)
