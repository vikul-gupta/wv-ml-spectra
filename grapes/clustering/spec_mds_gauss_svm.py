# Clusters three types of green grapes.
# Does dimensionality reduction, visualization of data, and classification
# using MDS, GMM, and SVM.

# Import all libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn import svm
import csv
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

csv_file = {0: '', 1: '_d1', 2: '_d2', 3: '_norm'}

for name in csv_file.values():
    # Get data from csv
    tmpdata = np.transpose(np.genfromtxt('grapes' + name  + '_w.csv', delimiter = ','))
    data = np.nan_to_num(tmpdata)

    # MDS model
    model = MDS(max_iter=200)
    np.set_printoptions(suppress=True)
    output = model.fit_transform(data)

    # Gaussian Mixture model
    g = mixture.GMM(n_components=2)
    g.fit(output)

    # Creation of labels
    labels = []
    for i in range(0,9):
        labels.append(1)
    for i in range(9,18):
        labels.append(2)
    for i in range(18,27):
        labels.append(3)

    # Apply support vector machine
    s = svm.SVC()
    s.fit(output, labels)

    # MCC calculation
    y_pred = s.predict(output)
    #print(labels)
    #print(y_pred)
    mcc = multimcc(labels,y_pred)
    print("MCC="+str(mcc))

    # Plot SVM contour
    h = .02  # step size in the mesh
    x_min, x_max = output[:, 0].min() - 1, output[:, 0].max() + 1
    y_min, y_max = output[:, 1].min() - 1, output[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = s.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot bunches 1, 2, and 3 for green grapes
    colors = {0: 'r', 1: 'g', 2: 'b'}
    for i in range(3):
        val = i*9
        output_i = output[val:val + 9]
        plt.scatter(output_i[:,0], output_i[:,1], color = colors[i], label = 'Bunch %d' % (i + 1))

    plt.title('TSNE for 3 types of green grapes')
    plt.legend()
    plt.show()
    savefig('graphs/plt_spec_mds_gmm_svm' + name  + '.png', bbox_inches='tight')

