#To Run
#Assumes:
#Model is named "model",
#Train and testing sets: "features_train", "y_train", "features_test", "y_test",
#Model output for test set is "y_model"

#should have same imports as Features.ipync
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
