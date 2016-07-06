# Clusters three types of grapes.
# Does dimensionality reduction and visualization of data
# using MDS and GMM.

# Import all libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import mixture
import csv
from matplotlib.pylab import savefig

colors = {0: 'r', 1: 'g', 2: 'b'}
csv_file = {0: '', 1: '_d1', 2: '_d2', 3: '_norm'}

for name in csv_file.values():
    # Get data from csv
    tmpdata = np.transpose(np.genfromtxt('grapes' + name  + '_w.csv', delimiter = ','))
    data = np.nan_to_num(tmpdata)

    # MDS model
    model = MDS(max_iter=200)
    np.set_printoptions(suppress=True)
    output = model.fit_transform(data)

    # GMM model
    g = mixture.GMM(n_components=2)
    g.fit(output)

    # Plot bunches 1, 2, and 3 for green grapes
    for i in range(3):
        val = i*9
        output_i = output[val:val + 9]
        plt.scatter(output_i[:,0], output_i[:,1], color = colors[i], label = 'Bunch %d' % (i + 1))

    plt.title('MDS and GMM for 3 types of green grapes')
    plt.legend()
    plt.show()
    savefig('graphs/plt_spec_mds_gauss' + name  + '.png', bbox_inches='tight')
