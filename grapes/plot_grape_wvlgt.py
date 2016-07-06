import numpy as np
import matplotlib.pyplot as plt

left_val = 0
tmpdata = np.transpose(np.genfromtxt('grapes_w.csv', delimiter = ','))
n, p = tmpdata.shape

# Find wavelengths
wl = []
for i in range(p):
    wl.append(234 + (0.27 * (i + left_val)))
#print (wl)

# Plot spectra
color = {0: 'r', 1: 'g', 2: 'b'}
for i in range(n):
    plt.plot(wl, tmpdata[i, :], color[int(i/9)])
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflection intensity')
plt.title('Spectra for Green Grapes: Basic')
plt.show()
