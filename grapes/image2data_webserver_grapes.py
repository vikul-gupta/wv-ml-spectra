# Script to take spectral image and obtain pre-processed data
# using the rasp_app web server. - WEBDRIVER (REFRESH THE PAGE) DOESN'T WORK

# Import all libraries
import os
import subprocess
from time import sleep

# Initializiation and moving to base directory
#base_dir = '~/wv2016/real_project/spectra_ML/grapes/jpgs/'
base_dir = './jpgs/grapes_old/'
rasp_dir = '~/wv2016/real_project/fruitoscopy/rasp_app/'
try:
    cmd = 'cd ' + base_dir
    subprocess.call(cmd, shell = True)
except KeyboardInterrupt:
    print ('good bye')

# Loop through all files in the directory
files = sorted(os.listdir(base_dir))
i = 0
for j in files:
    i += 1
    # Copy file from base directory and name it 'source.jpg'
    try:
        cmd = 'cp jpgs/grapes_old/' + j + ' ' + rasp_dir + '; cd ' + rasp_dir + '; mv ' + j + ' source.jpg; cd ../../spectra_ML/grapes'
        subprocess.call(cmd, shell = True)
    except KeyboardInterrupt:
        print ('good bye')
    sleep(0.9)

# Move csv file to grapes directory
try:
    cmd = 'mv ' + rasp_dir + 'static/spectrum.csv .'
    subprocess.call(cmd, shell = True)
except KeyboardInterrupt:
    print ('good bye')

