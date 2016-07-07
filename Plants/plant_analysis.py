from sklearn import preprocessing
from scipy.signal import find_peaks_cwt
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

# Calibrate the data
def calibrate(data, ref):
    # Divide data by reference
    cal = []
    for i in range(data.shape[0]):
        cal.append(data[i]/ref[int(i/12)])
    # Normalize data between 0 and 1
    cal = np.array(cal)
    norm = cal
    return norm

# Get data as nparray
def getdata(name, ref_val):
    if ref_val:
        #tmpdata = np.genfromtxt(name, delimiter = ',')[750:1601] # chlorophyll
        tmpdata = np.genfromtxt(name, delimiter = ',')[2400:2780] # water potential
    else:
        tmpdata = np.genfromtxt(name, delimiter = ',')/255
    data = np.nan_to_num(tmpdata)
    return data

# Find peak b/w wv 600nm and 650nm for chlorophyll; 930 and 960 for water potential
def find_peak(cdata, left_val, wl):
    index = []
    index_x = []
    for signal in cdata:
        #index.append(find_peaks_cwt(signal, noise_perc = 60, widths = np.array([50.])))
        id_i = (600 - 227)/0.26 - left_val
        id_f = (650 - 227)/0.26 - left_val
        max_val = np.max(signal[id_i:id_f])
        index.append(max_val)
        x_wl = 227 + (0.26 * (np.where(signal[id_i:id_f] == max_val)[0][0] + left_val))
        index_x.append(x_wl)
    #print(index_x)
    return index, index_x

# Find wavelengths
def wavelength(left_val):
    wl = []
    for i in range(p):
        wl.append(227 + (0.26 * (i + left_val)))
    #print (wl)
    return wl

# Plot normalized data spectra
def plot_spectra(cdata, n, colors):
    sum_spec = np.zeros((5, cdata.shape[1]))
    for i in range(n):
        plt.plot(wl, cdata[i, :], colors[int(i/16)], alpha = 0.2, linewidth = 1)
        sum_spec[int(i/16)] += cdata[i]
    mean_spec = sum_spec/16
    for i in range(5):
        plt.plot(wl, mean_spec[i, :], colors[i], linewidth = 2.5, label = 'Plant %d' % (i+1))

# Calculate and plot means of peaks
def plot_mean(index, index_x, colors):
    mean_y_vals = [np.mean(index[16*i : 16*(i + 1)]) for i in range(5)]
    mean_x_vals = [np.mean(index_x[16*i : 16*(i + 1)]) for i in range(5)]
    for i in range(5):
        plt.scatter(mean_x_vals[i], mean_y_vals[i], color = colors[i], label = 'Plant %d' % (i + 1))

# Plot peak values
def plot_index(index, index_x, n, colors):
    print (colors)
    for i in range(n):
        plt.scatter(index_x, index, color = colors[int(i/16)])

# Initialize stuff
name = 'chlorophyll_2'
left_val = 900     # for chlorophyll
#left_val = 2400    # for water potential
colors = {0: 'r', 1: 'y', 2: 'g', 3: 'b', 4: 'k'}
cdata = getdata('spectrum_' + name + '.csv', False)
n, p = cdata.shape
ref = getdata('white_ref.dat', True)

# Calculate values
#cdata = calibrate(data, ref)
wl = wavelength(left_val)
index, index_x = find_peak(cdata, left_val, wl)

# Plot stuff
plot_spectra(cdata, n, colors)
#plot_mean(index, index_x, colors)
#plot_index(index, index_x, n, colors)

# Compare peak value of all 5 variables (1, 2, 3, 4, 5).


# Plot formatting
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflection intensity')
#plt.title('Spectra for ' + name.title())
plt.legend()
plt.show()

