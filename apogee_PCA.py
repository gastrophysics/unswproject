# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:29:42 2016

@author: john
"""

#PCA on full (or maybe disk only) apogee set in prep. for k-means and others

import numpy as np 
import sklearn.decomposition as dr
from astropy.io import fits
import pylab as plt
import pickle as cp

#load CANNON data

data_labels = ["FE_H",
               "AL_H", "CA_H", "C_H", "K_H",  "MG_H", "MN_H", "NA_H",
               "NI_H", "N_H",  "O_H", "SI_H", "S_H",  "TI_H", "V_H"]

hdulist = fits.open('data/results-unregularized-matched.fits')
cols = hdulist[1].columns
tbdata = hdulist[1].data
#maybe add in rejection of non disks
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

#PCA the stuffs without transforming the data against iron
#pca = dr.PCA()
#
#pca_data = pca.fit(data)
#vectors = pca.components_
#vector_variance = pca.explained_variance_ratio_
#proper_index = range(len(vector_variance)+1)
#proper_index = proper_index[1:]
#plt.plot(proper_index, vector_variance.cumsum(), 'b')
#plt.xlim([1, len(proper_index)])
#print(data.shape)
#PCA the stuffs with transforming the data against iron, ends up identical
fe_data = np.vstack((data.T[i] - data.T[0]) for i in range(len(data_labels))).T
fe_data.T[0] = data.T[0]

pca = dr.PCA()
pca_data = pca.fit(fe_data)
vectors = pca.components_
vector_variance = pca.explained_variance_ratio_
plt.plot(range(1, 16), vector_variance.cumsum())
plt.xlim(1)

#select cutoff with components at above 90% (roughly 7)
cutoff = 0
keep = 0
for variance in vector_variance:
    cutoff += variance
    keep += 1
    if cutoff > 0.9:
        break
print("we will keep", keep, "principal components")

#transform the data
reduce = dr.PCA(n_components = keep)
reduced_data = reduce.fit_transform(fe_data)
print("transforming the data yields a new set with shape:", reduced_data.shape)

#pickle the transformed data
fd = open("data/Apogee_PCA_90.pkl", "wb")
cp.dump(reduced_data, fd)
fd.close()

"""
more code to check other components roughly
"""
#change data to disk only
#mask = ((tbdata.field("TEFF_ASPCAP") > 3500.) *
#        (tbdata.field("TEFF_ASPCAP") < 5500.) *
#        (tbdata.field("GLAT") < 20) *
#        (tbdata.field("GLAT") > -20) *
#        (tbdata.field("LOGG_ASPCAP") > 0.) *
#        (tbdata.field("LOGG_ASPCAP") < 3.9)) # MKN recommendation
#metadata_check_labels = ["RA", "DEC", "TEFF_ASPCAP", "LOGG_ASPCAP"]
#metadata_check = np.vstack((tbdata.field(label) for label in metadata_check_labels)).T
#mask *= np.all(np.isfinite(metadata_check), axis=1)
#data = np.vstack((tbdata.field(label) for label in data_labels)).T
#mask *= np.all(np.isfinite(data), axis=1)
#data = data[mask]
#pca = dr.PCA()
#pca_data = pca.fit(data)
#vectors = pca.components_
#vector_variance = pca.explained_variance_ratio_
#proper_index = range(len(vector_variance)+1)
#proper_index = proper_index[1:]
#plt.plot(proper_index, vector_variance.cumsum(), 'r')
#print(data.shape)
#
##change data to anything but disk
#mask = ((tbdata.field("TEFF_ASPCAP") > 3500.) *
#        (tbdata.field("TEFF_ASPCAP") < 5500.) *
#        ((tbdata.field("GLAT") > 20) + (tbdata.field("GLAT") < -20)) *
#        (tbdata.field("LOGG_ASPCAP") > 0.) *
#        (tbdata.field("LOGG_ASPCAP") < 3.9)) # MKN recommendation
#metadata_check_labels = ["RA", "DEC", "TEFF_ASPCAP", "LOGG_ASPCAP"]
#metadata_check = np.vstack((tbdata.field(label) for label in metadata_check_labels)).T
#mask *= np.all(np.isfinite(metadata_check), axis=1)
#data = np.vstack((tbdata.field(label) for label in data_labels)).T
#mask *= np.all(np.isfinite(data), axis=1)
#data = data[mask]
#pca = dr.PCA()
#pca_data = pca.fit(data)
#vectors = pca.components_
#vector_variance = pca.explained_variance_ratio_
#proper_index = range(len(vector_variance)+1)
#proper_index = proper_index[1:]
#plt.plot(proper_index, vector_variance.cumsum(), 'g')
#print(data.shape)

#we find an ok spread with these "loose" components