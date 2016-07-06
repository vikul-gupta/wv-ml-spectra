# Clusters three types of green grapes.
# Does dimensionality reduction and visualization of data
# using MDS.

# Import all libraries
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import csv
from matplotlib.pylab import savefig

csv_file = {0: '', 1: '_d1', 2: '_d2', 3: '_norm'}

for name in csv_file.values():
    # Get data from csv
    tmpdata = np.transpose(np.genfromtxt('grapes' + name  + '_w.csv', delimiter = ','))
    data = np.nan_to_num(tmpdata)

    # Apply Multidimensional Scaling
    model = MDS(max_iter=200)
    np.set_printoptions(suppress=True)
    output = model.fit_transform(data)

    # Plot bunches 1, 2, and 3 for green grapes
    colors = {0: 'r', 1: 'g', 2: 'b'}
    for i in range(3):
        val = i*9
        output_i = output[val:val + 9]
        plt.scatter(output_i[:,0], output_i[:,1], color = colors[i], label = 'Bunch %d' % (i + 1))

    plt.title('Multidimensional Scaling (MDS)')
    plt.legend()
    plt.show()
    savefig('graphs/plt_spec_mds' + name  + '.png', bbox_inches='tight')

