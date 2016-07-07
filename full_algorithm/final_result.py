from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy import stats
from sklearn import svm
from sklearn.externals import joblib
import json

#MODULO 1
#LABELS SEPARATED FROM THE MATRIX

#loading data
tmpdata = np.genfromtxt('exit.csv', delimiter=' ')
X = np.nan_to_num(tmpdata)


# Creation of labels
tmpdata = np.genfromtxt('labels.csv', delimiter=' ')
y = np.nan_to_num(tmpdata)

with open('parameters.json') as data_file:
    dic = json.load(data_file)

model = dic['model']

bin_num = dic['num_bins']
parameter = dic['parameter']

#MODULO2
#PROCESS THE DATA(BINNING)

#binning model matrix,binned matrix==stat 
stat, bin_edges, binnum = stats.binned_statistic(range(X.shape[1]), X, 'median', bins=int(bin_num))

#MODULO3
#APPLY THE MODEL AND PRINT THE RESULT

if model == 'svm.LinearSVC()':
    clf=svm.LinearSVC(C=parameter)
if model == 'RandomForestClassifier()': 
    clf=RandomForestClassifier(n_estimators=parameters,n_jobs=-1)
if model == 'LinearDiscriminantAnlysis()': 
    clf=LinearDiscriminantAnalysis()

out = clf.fit(stat,y)
output = clf.fit_transform(stat, y)

'''
# Plot SVM contour
# Can't plot it because it is 160 dimensions
h = .02  # step size in the mesh
x_min, x_max = output[:, 0].min() - 1, output[:, 0].max() + 1
y_min, y_max = output[:, 1].min() - 1, output[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
arr_conc = np.c_[xx.ravel(), yy.ravel()] # concatenate two arrays together
#print (arr_conc)
#print (arr_conc.shape)
Z = clf.predict(arr_conc)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot bunches 1, 2, and 3 for green grapes
colors = {0: 'r', 1: 'g', 2: 'b'}
for i in range(3):
    val = i*9
    output_i = output[val:val + 9]
    plt.scatter(output_i[:,0], output_i[:,1], color = colors[i], label = 'Bunch %d' % (i + 1))

plt.title('Best model for 3 types of green grapes')
plt.legend()
plt.show()
'''
joblib.dump(clf ,"model.pkl")
#a_result=joblib.load("model.pkl")

