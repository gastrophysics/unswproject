# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:21:46 2016

@author: john
"""


import os
import pickle as cp
import numpy as np 
import sklearn.cluster as cl
from astropy.io import fits
import pylab as plt

#abundances available:
data_labels = ["FE_H",
               "AL_H", "CA_H", "C_H", "K_H",  "MG_H", "MN_H", "NA_H",
               "NI_H", "N_H",  "O_H", "SI_H", "S_H",  "TI_H", "V_H"]
#let's forget about the ones we don't care about so much, we could also choose
#to make small deviations in these as flags for non candidates (or rule out any candidates later)
#for x in ["SI_H", "CA_H", "SC_H", "TI_H", "V_H", "CR_H", "MN_H", "NI_H"]:
#    if x in data_labels:
#        data_labels.remove(x)
#retrieve survey data: APOGEE and mask
hdulist = fits.open('data/results-unregularized-matched.fits')
cols = hdulist[1].columns
tbdata = hdulist[1].data
mask = ((tbdata.field("TEFF_ASPCAP") > 3500.) *
        (tbdata.field("TEFF_ASPCAP") < 5500.) *
        (tbdata.field("LOGG_ASPCAP") > 0.) *
        (tbdata.field("LOGG_ASPCAP") < 3.9)) # MKN recommendation
metadata_check_labels = ["RA", "DEC", "TEFF_ASPCAP", "LOGG_ASPCAP"]
metadata_check = np.vstack((tbdata.field(label) for label in metadata_check_labels)).T
mask *= np.all(np.isfinite(metadata_check), axis=1)
data = np.vstack((tbdata.field(label) for label in data_labels)).T
mask *= np.all(np.isfinite(data), axis=1)
data = data[mask]

#we will have to grab some metadata ID, GLON/GLAT and whatever else
#is interesting to us after analysing the data
metadata_labels = ["RA", "DEC", "GLON", "GLAT", "VHELIO_AVG", "TEFF_ASPCAP", "LOGG_ASPCAP", "FIELD", "APOGEE_ID"]
metadata = np.vstack((tbdata.field(label) for label in metadata_labels)).T
metadata = metadata[mask]

#scale_dict = {"FE_H": 0.0191707168068, # all from AC
#              "AL_H": 0.0549037045265,
#              "CA_H": 0.0426365845422,
#              "C_H": 0.0405909985963,
#              "K_H": 0.0680897262727,
#              "MG_H": 0.0324021951804,
#              "MN_H": 0.0410348634747,
#              "NA_H": 0.111044350016,
#              "NI_H": 0.03438986215,
#              "N_H": 0.0440559383568,
#              "O_H": 0.037015877736,
#              "SI_H": 0.0407894206516,
#              "S_H": 0.0543424861906,
#              "TI_H": 0.0718311106542,
#              "V_H": 0.146438163035, }

km = cl.KMeans(n_clusters = 256, random_state=42, n_init=32)
scales = np.array([scale_dict[label] for label in data_labels])
clusters = km.fit_predict(data / scales[None, :])
centers = km.cluster_centers_.copy()



#plot apogee, plot the last cluster
#plt.plot(metadata.T[2], metadata.T[3], color = '0.5', marker = ',', ls = 'None')
#plt.plot(outliers.T[2], outliers.T[3], 'bo')

