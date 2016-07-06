import numpy as np
import matplotlib.pyplot as plt

left_val = 900     # chlorophyll
#left_val = 2400    # water_1
name = 'chlorophyll_2'
tmpdata = np.genfromtxt('spectrum_' + name + '.csv', delimiter = ',')
n, p = tmpdata.shape

# Find wavelengths
wl = []
for i in range(p):
    wl.append(234 + (0.27 * (i + left_val)))
#print (wl)

# Plot spectra
#color = {0: 'r', 1: 'g', 2: 'b'}
for i in range(n):
    plt.plot(wl, tmpdata[i, :], 'b')
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflection intensity')
plt.title('Spectra for ' + name.title())
plt.show()
