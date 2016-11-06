# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:45:29 2016

@author: john
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import sklearn.decomposition as dr



#open up component files
bulge = open('data/stars-bulge-2.dat')
halo = open('data/stars-halo-2.dat')
thick = open('data/stars-thickdisk-2.dat')
thin = open('data/stars-thindisk-2.dat')
clusters = open('data/stars-clusters.dat')

cols = ["FE_H",
          "AL_FE", "CA_FE", "C_FE", "K_FE",  "MG_FE", "MN_FE", "NA_FE",
          "NI_FE", "N_FE",  "O_FE", "SI_FE", "S_FE",  "TI_FE", "V_FE"]

meta_labels = ["APOGEE_ID"]
IDs = []
#fix data type
components = [bulge, halo, thick, thin, clusters]
comp_abundances = []
for component in components:
    c = component.readlines()
    abundances = ascii.read(c)
    comp_abundances.append(abundances)
#reduce to just the abundances and IDs
components = []
for component in comp_abundances:
    c = np.vstack((component.field(col) for col in cols)).T
    ID = np.vstack((component.field(label) for label in meta_labels)).T
    components.append(c)
    IDs.append(ID)
#
n = components[0].shape[1]

#perform PCA on each and graph

#mask out the 27 nan rows occuring in clusters (thats everything)
mask = np.alltrue(np.isfinite(components[4]),1)
components[4] = components[4][mask]

plot_left = []
for component in components:
    pca = dr.PCA()
    pca_data = pca.fit(component)
    vector_variance = pca.explained_variance_ratio_
    plot_left.append(vector_variance.cumsum())
#    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#    ax1.plot(range(1, n+1), vector_variance.cumsum(), colours[0])
#    colours = colours[1:]
#plt.xlim([1, n])
    
#convert to H Basis
plot_right = []
for component in components:
    fe_vector = component.T[0] #iron column
    fe_component = np.vstack((component.T[i].T + fe_vector.T) for i in range(1, 15))
    fe_component = np.vstack((fe_vector.T, fe_component))
    fe_component = fe_component.T
    pca = dr.PCA()
    pca_data = pca.fit(fe_component)
    vector_variance = pca.explained_variance_ratio_
    plot_right.append(vector_variance.cumsum())

    
#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)    
colours = ['k', 'm', 'b', 'g', 'r']
for plot in plot_left:
    plot = np.hstack((0, plot))
    plt.plot(range(0, n+1), 100*plot, colours[0])
    colours = colours[1:]
colours = ['k', 'm', 'b', 'g', 'r']
#for plot in plot_left:
#    plot = np.hstack((0, plot))
#    plt.plot(range(0, n+1), 100*plot, colours[0])
#    colours = colours[1:]
#prettify the plots
plt.xlim([0, 15])
plt.axhline(y = 85, linestyle = 'dashed')
plt.axhline(y = 95, linestyle = 'dashed')
#plt.xlim([0, 15])
#ax2.axhline(y = 85, linestyle = 'dashed')
#ax2.axhline(y = 95, linestyle = 'dashed')
plt.ylim([0, 100])
#ax2.set_ylim([0, 100]) 
plt.xlabel("Component Number") 
#ax2.set_xlabel("Component Number")
plt.ylabel("Variance Contribution (Cumulative Percentage)") 
plt.title("PCA on [X/H] Basis")

