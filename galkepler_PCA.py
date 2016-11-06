# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:49:58 2016

@author: john
"""

import numpy as np 
import sklearn.decomposition as dr
from astropy.io import fits
import pylab as plt
from sklearn.preprocessing import Imputer

#abundances available:
label_tail = "_abund_sme"
data_labels = ["feh_sme",
               "li", "c", "o", "mg",  "al", "si", "k",
               "ca", "sc",  "ti", "v", "cr",  "mn", "co", "ni", "cu", "zn", 
               "y", "ba", "la", "nd", "eu", "rb"]
new_labels =[]
for label in data_labels:
    new_labels.append(label + label_tail)
new_labels[0] = "feh_sme" #because fe_abund_sme is wrong
data_labels = new_labels
hdulist = fits.open('data/sobject_iraf_k2.fits')
cols = hdulist[1].columns
tbdata = hdulist[1].data
#mask =  ((tbdata.field("feh_sme") < 0.2) *
#        (tbdata.field("feh_sme") > -0.2))
data = np.vstack((tbdata.field(label) for label in data_labels)).T
#mask = np.all(np.isfinite(data), axis=1) #damn it every object is busted
#data = data[mask]
#the data is filled with bloody nans so impute the mean
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data)
imputed_data = imp.transform(data)

pca = dr.PCA()

pca_data = pca.fit(imputed_data)
vectors = pca.components_
vector_variance = pca.explained_variance_ratio_
proper_index = range(len(vector_variance)+1)
proper_index = proper_index[1:]
plt.plot(proper_index, vector_variance.cumsum(), 'b')
plt.xlim([1, len(proper_index)])
print(imputed_data.shape)