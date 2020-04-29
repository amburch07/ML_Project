import numpy as np
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import preprocessing
from sklearn.svm import SVC
from detect_peaks import detect_peaks
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import kurtosis
from scipy.stats import skew
from pandas import read_csv
from sklearn.metrics import precision_score

fs = 50;
g = 9.80665;

#Load datasets for test and training data
def loadTAData(isTrain=True):
    if (isTrain):
        #load xyz total accel
        ax = np.loadtxt('HARDataset/train/Inertial_Signals/total_acc_x_train.txt')
        ay = np.loadtxt('HARDataset/train/Inertial_Signals/total_acc_y_train.txt')
        az = np.loadtxt('HARDataset/train/Inertial_Signals/total_acc_z_train.txt')

        # Load train labels
        y = np.loadtxt('HARDataset/train/y_train.txt')
        
    else:
        ax = np.loadtxt('HARDataset/test/Inertial_Signals/total_acc_x_test.txt')
        ay = np.loadtxt('HARDataset/test/Inertial_Signals/total_acc_y_test.txt')
        az = np.loadtxt('HARDataset/test/Inertial_Signals/total_acc_z_test.txt')
        #xyz accel data (stacked shoulder to shoulder and pickled)
        #a_test = np.load('HARDataset/test/Inertial Signals/total_acc_xyz_test.npy')
        #ax = a_test[:,:128]; ay = a_test[:,128:256]; az = a_test[:,256:]

        #Load test labels
        y = np.loadtxt('HARDataset/test/y_test.txt')
           
    return ax, ay, az, y


# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = read_csv(prefix + name, header=None, delim_whitespace=True).values
        loaded.append(data)
    loaded = np.dstack(loaded) # stack group so that features are the 3rd dimension
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, which, prefix=''):
    filepath = prefix + group + '/Inertial_Signals/'
    filenames = list() # load all files as a single array
    if (which=='BA'): # body acceleration
        filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    elif (which=='BG'): # body gyroscope
        filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    else:
        print("Error");
    # load input data
    X = load_group(filenames, filepath)
    return X

def loadBAGData(isTrain=True, which='BA', prefix='HARDataset'):
    if (isTrain):
        #load xyz body accel/gyro for train (stacked shoulder to shoulder)
        X = load_dataset_group('train', which, prefix + '/') #training set
    else:
        #load xyz body accel/gyro for test 
        X = load_dataset_group('test', which, prefix + '/') #test set
    #body accel or gyro
    tx = X[:,:,0]
    ty = X[:,:,1]
    tz = X[:,:,2]
               
    return tx, ty, tz


def splitVec(a):
    a_len = int(np.shape(a)[1]/4)
    
    ax = a[:,:a_len]
    ay = a[:,a_len:2*a_len]
    az = a[:,2*a_len:3*a_len]
    am = a[:,3*a_len:]
    
    return ax, ay, az, am


#Normalize data or features
def normData(a_train, a_test,withm=False):
    if (withm):
        a_len = int(np.shape(a_train)[1]/4)
        m_scaler = preprocessing.StandardScaler().fit(a_train[:,3*a_len:])
        m_scaler.transform(a_train[:,3*a_len:])
    else:
        a_len = int(np.shape(a_train)[1]/3)
    
    #scalars to normalize
    x_scaler = preprocessing.StandardScaler().fit(a_train[:,:a_len])
    y_scaler = preprocessing.StandardScaler().fit(a_train[:,a_len:2*a_len])
    z_scaler = preprocessing.StandardScaler().fit(a_train[:,2*a_len:3*a_len])
    
    #normalize training
    x_scaler.transform(a_train[:,:a_len])
    y_scaler.transform(a_train[:,a_len:2*a_len])
    z_scaler.transform(a_train[:,2*a_len:3*a_len])
    
    #normalize testing
    if (withm):
        a_len = int(np.shape(a_test)[1]/4)
        m_scaler.transform(a_test[:,3*a_len:])
    else:
        a_len = int(np.shape(a_train)[1]/3)
    x_scaler.transform(a_test[:,:a_len])
    y_scaler.transform(a_test[:,a_len:2*a_len])
    z_scaler.transform(a_test[:,2*a_len:3*a_len])
    
    return a_train, a_test


#Calculate the magnitude of a set of features
def calcMag(ax, ay, az):
    am = np.sqrt(np.add(np.add(np.square(ax), np.square(ay)), np.square(az)))
    amdt = am - np.mean(am,axis=0)
    amdt = amdt / np.std(am,axis=0)
    return amdt

#Convert set of TD features to FD
def convertFD(tx, ty, tz):
    freq, fx = signal.welch(tx, fs,nperseg = 128)
    freq, fy = signal.welch(ty, fs,nperseg = 128)
    freq, fz = signal.welch(tz, fs,nperseg = 128)    
    fm = calcMag(fx,fy,fz)
    return freq, fx, fy, fz, fm


#Calculate TD or FD Means & Stds
def muStds(a):
    ax, ay, az, am = splitVec(a)
    
    axMu = np.mean(ax,axis=1)
    ayMu = np.mean(ay,axis=1)
    azMu = np.mean(az,axis=1)
    amMu = np.mean(am,axis=1)
    #mus = np.vstack((axMu, ayMu, azMu, amMu)).T #uncomment if you want magnitude
    mus = np.vstack((axMu, ayMu, azMu)).T
    
    axStd = np.std(ax,axis=1)
    ayStd = np.std(ay,axis=1)
    azStd = np.std(az,axis=1)
    amStd = np.std(am,axis=1)
    #stds = np.vstack((axStd, ayStd, azStd, amStd)).T #uncomment if you want magnitude
    stds = np.vstack((axStd, ayStd, azStd)).T
    
    return mus, stds 




#Calculate TD or FD Skew & Kurtosis
def skewKurt(a):
    ax, ay, az, am = splitVec(a)
    
    axSk = skew(ax,axis=1)
    aySk = skew(ay,axis=1)
    azSk = skew(az,axis=1)
    amSk = skew(am,axis=1)
    #skews = np.vstack((axSk, aySk, azSk, amSk)).T #uncomment if you want magnitude
    skews = np.vstack((axSk, aySk, azSk)).T
    
    axKurt = kurtosis(ax,axis=1)
    ayKurt = kurtosis(ay,axis=1)
    azKurt = kurtosis(az,axis=1)
    amKurt = kurtosis(am,axis=1)
    #kurts = np.vstack((axKurt, ayKurt, azKurt, amKurt)).T #uncomment if you want magnitude
    kurts = np.vstack((axKurt, ayKurt, azKurt)).T
    
    return skews, kurts



#Find TD or FD Peaks - Number of Peaks, and Top n Largest Peaks
def findPeaks(a,n):
    mpd = 80
    mph =4
    edge='rising'
    
    a_rows = np.shape(a)[0]
    ax,ay,az,am = splitVec(a)
    
    a_p = np.zeros((a_rows,4*n));
    a_np = np.zeros((a_rows,4*n));
    for i in range (a_rows):
        fx_peaks = detect_peaks(ax[i,:],edge=edge, show=False)
        fy_peaks = detect_peaks(ay[i,:],edge=edge, show=False)
        fz_peaks = detect_peaks(az[i,:],edge=edge, show=False)
        fm_peaks = detect_peaks(am[i,:],edge=edge, show=False)
        
        a_p[i,:n] = np.sort(ax[i,fx_peaks])[-1*n:]
        a_p[i,n:2*n] = np.sort(ay[i,fy_peaks])[-1*n:]
        a_p[i,2*n:3*n] = np.sort(az[i,fz_peaks])[-1*n:]
        a_p[i,3*n:] = np.sort(am[i,fm_peaks])[-1*n:]
        
        a_np[i,:n] = len(fx_peaks);
        a_np[i,n:2*n] = len(fy_peaks);
        a_np[i,2*n:3*n] = len(fz_peaks);
        a_np[i,3*n:] = len(fm_peaks);

    return a_p, a_np #uncomment if you want magnitude
    # return a_p[:,:3*n], a_np[:,:3*n]


def makePipeline(features_train, y_train, features_test, y_test, degree=1):
    #make pipeline
    #polynomials up to degree 2
    #then SVC with RBF kernal
    model = make_pipeline(PolynomialFeatures(degree=degree),
                          SVC())
    model.fit(features_train, y_train) #train
    y_model = model.predict(features_test) #test results

    #timing
    get_ipython().run_line_magic('time', 'model.fit(features_train, y_train)')
    get_ipython().run_line_magic('time', 'y_pred = model.predict(features_test)')

    #accuracy
    print("Train accuracy is %.2f " % (model.score(features_train, y_train)*100))
    print("Test accuracy is %.2f " % (model.score(features_test, y_test)*100))

    #accuracy and confusion matrix
    acc = accuracy_score(y_test, y_model)
    mat = confusion_matrix(y_test, y_model)

    #AUC/Precision Score
    ps = precision_score(y_test, y_model, average="macro")
    print("AUC: %.3f" %ps)
    
    plt.figure()
    sns.heatmap(mat, square=True, annot=True, cbar=False) #, cmap='YlGnBu', flag, YlGnBu, jet
    plt.xlabel('predicted value')
    plt.ylabel('true value');
    
    return model