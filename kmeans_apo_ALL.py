#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:30:00 2016

@author: john
"""

import numpy as np 
import sklearn.cluster as cl
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
print("data is shape: ",data.shape)
#convert to the iron basis
fe_data = np.vstack((data.T[i] - data.T[0]) for i in range(len(data_labels))).T
fe_data.T[0] = data.T[0]

print("fe basis data is shape: ",fe_data.shape)

#reestablish metadata that we actually need
metadata_labels = ["RA", "DEC", "PMRA", "PMDEC", "GLON", "GLAT", "VHELIO_AVG", "TEFF_ASPCAP", "LOGG_ASPCAP", "FIELD", "APOGEE_ID"]
metadata = np.vstack((tbdata.field(label) for label in metadata_labels)).T
metadata = metadata[mask]

print("metadata is shape: ",metadata.shape)

#now we want some kmeans up in here
#K = 256
#print("running kmeans, k = 256 on whole set...")
#km = cl.KMeans(n_clusters = K, random_state=42, n_init=32) #the algorithm
#clusters = km.fit_predict(data) #the important returned vale
#centers = km.cluster_centers_.copy() #needed for variance
#
##compute the density so we can retrieve the densest
#print("kmeans is done, we now compute density")
##hogg style below (not sure how reliable this is):
#N, D = data.shape
#sizes = np.zeros(K).astype(int)
#logdets = np.zeros(K)
#fehs = np.zeros(K)
#for k in range(K):
#    I = (clusters == k)
#    sizes[k] = np.sum(I)
#    subdata = data[I,:] - (np.mean(data[I], axis=0))[None, :]
#    if sizes[k] > (D + 1):
#        variance = np.sum(subdata[:,:,None] * subdata[:,None,:], axis=0) / (sizes[k] - 1.)
#        s, logdets[k] = np.linalg.slogdet(variance)
#        assert s > 0
#    else:
#        logdets[k] = -np.Inf
#    fehs[k] = np.mean(data[I, 0])
#densities = sizes * np.exp(-0.5 * logdets)
#
#
##we can plot this shit
#plt.loglog(sizes, densities, '.')
#plt.xlabel("Cluster Size")
#plt.ylabel("Cluster Density")
#it is quite similar to hogg et al
#remember there is no scaling for our space
#remember that we have not put it in the fe basis space