# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:51:42 2016

@author: john
"""

import pickle as cp
import numpy as np 
import sklearn.cluster as cl
import pylab as plt


#open up the file of interest: apo_pca_90
fd = open("data/Apogee_PCA_90.pkl", "rb")
data = cp.load(fd)
fd.close()

print("we have loaded up the file with shape:", data.shape)

print("running kmeans on reduced apogee data...")
km = cl.KMeans(n_clusters = 256, random_state=42, n_init=32)
clusters = km.fit_predict(data)
centers = km.cluster_centers_.copy()

#pickle the kmeans data
fd = open("data/km_Apogee_PCA_90.pkl", "wb")
cp.dump(clusters, fd)
fd.close()