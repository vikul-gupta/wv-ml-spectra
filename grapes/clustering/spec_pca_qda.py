# Clusters three types of green grapes.
# Does dimensionality reduction, visualization of data, and classification 
# using PCA and QDA.

# Import all libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import matthews_corrcoef
import csv
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib.pylab import savefig

def multimcc(t,p, classes=None):
    """ Matthews Correlation Coefficient for multiclass
    :Parameters:
        t : 1d array_like object integer 
          target values
        p : 1d array_like object integer 
          predicted values
        classes: 1d array_like object integer containing 
          all possible classes
    
    :Returns:
        MCC : float, in range [-1.0, 1.0]
    """

    # Cast to integer
    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)
    
    
    # Get the classes
    if classes is None:
        classes = np.unique(tarr)
    
    nt = tarr.shape[0]
    nc = classes.shape[0]
    
    # Check dimension of the two array
    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")

    # Initialize X and Y matrices
    X = np.zeros((nt, nc))
    Y = np.zeros((nt, nc))

    # Fill the matrices 
    for i,c in enumerate(classes):
        yidx = np.where(tarr==c)
        xidx = np.where(parr==c)

        X[xidx,i] = 1
        Y[yidx,i] = 1

    # Compute the denominator
    denom = cov(X,X) * cov(Y,Y)
    denom = np.sqrt(denom)
    
    if denom == 0:
        # If all samples assigned to one class return 0
        return 0
    else:
        num = cov(X,Y)
        return num / denom


def confusion_matrix(t, p):
    """ Compute the multiclass confusion matrix
    :Parameters:
        t : 1d array_like object integer (-1/+1)
          target values
        p : 1d array_like object integer (-1/+1)
          predicted values
    
    :Returns:
        MCC : float, in range [-1.0, 1.0]
    """

    # Read true and predicted classes
    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)
    
    # Get the classes
    classes = np.unique(tarr)

    # Get dimension of the arrays
    nt = tarr.shape[0]
    nc = classes.shape[0]

    # Check dimensions should match between true and predicted
    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")

    # Initialize Confusion Matrix C
    C = np.zeros((nc, nc))

    # Fill the confusion matrix
    for i in xrange(nt):
        ct = np.where(classes == tarr[i])[0]
        cp = np.where(classes == parr[i])[0]
        C[ct, cp] += 1

    # return the Confusion matrix and the classes
    return C, classes

def cov(x,y):
    nt = x.shape[0]
    
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xxm = x - xm
    yym = y - ym
    
    tmp = np.sum(xxm * yym, axis=1)
    ss = tmp.sum()

    return ss/nt

num_class = 3
colors = {0: 'r', 1: 'g', 2: 'b'}
csv_file = {0: '', 1: '_d1', 2: '_d2', 3: '_norm'}

for name in csv_file.values():
    # Getting data from csv
    tmpdata = np.transpose(np.genfromtxt('grapes' + name  + '_w.csv', delimiter = ','))
    data = np.nan_to_num(tmpdata)

    # PCA model
    pca = PCA(n_components=5)
    comps = pca.fit_transform(data) 
    #plt.plot(pca.components_.reshape((2,data.shape[0],data.shape[1])))
    #plt.plot(pca.explained_variance_, linewidth=2)
    #plt.title('Principal Component Analysis (PCA) Feature Assessment')

    # Creation of actual labels
    labels = []
    
    for i in range(0,9):
        labels.append(1)
    for i in range(9,18):
        labels.append(2)
    for i in range(18, 27):
        labels.append(3)
    '''
    # Creation of random labels
    for i in range(0,27):
        labels.append(int(random.random() * 3) + 1)
    print (labels)
    '''
    # QDA model
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(comps, labels)

    # MCC Calculation
    y_pred = qda.predict(comps)
    #print(labels)
    #print(y_pred)
    mcc = multimcc(labels,y_pred)
    print("MCC="+str(mcc))

    '''
    # Plotting QDA contour
    nx, ny = 200, 100
    x_min, x_max = np.amin(comps[:,0]), np.amax(comps[:,0])
    y_min, y_max = np.amin(comps[:,1]), np.amax(comps[:,1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))
    Z = qda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, Z, [0.5], linewidths=5, colors = 'k', linestyles = 'dashed')
    '''

    # Plotting QDA means
    for i in range(num_class):
        plt.plot(qda.means_[i][0], qda.means_[i][1], 'o', color = colors[i], markersize = 10)
    plt.title('PCA with QDA')

    # Plot bunches 1, 2, and 3 for green grapes
    for i in range(num_class):
        val = i*9
        comps_i = comps[val:val + 9]
        plt.scatter(comps_i[:,0], comps_i[:,1], color = colors[i], label = 'Bunch %d' % (i + 1))

    plt.title('PCA and QDA for 3 types of green grapes')
    plt.legend()
    plt.show()
    savefig('graphs/plt_spec_pca_qda' + name  + '.png', bbox_inches='tight')

