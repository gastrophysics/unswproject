#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:11:50 2016

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.io import ascii
from astropy.io import fits
#import sklearn.decomposition as dr
import sklearn.cluster as cl
import pickle as cp

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

disk_IDs = np.vstack((IDs[2], IDs[3]))
disk_IDs = disk_IDs.T[0]
disk_only = np.vstack((components[2], components[3]))

#refine all stars from 98489 to 98462
mask = np.alltrue(np.isfinite(components[4]),1)
all_apo_stars = components[4][mask]

def run_dbscan(mnpts, epsi):
    try:
        fd = open("data/dbscans/apo_fe_db_" + str(mnpts) + "_" + str(epsi) + ".pkl", "rb")
        clusters = cp.load(fd)
        fd.close()
        print("loaded clusters from file")
    except:
        print("no clusters file so perform DBSCAN")
        print("running DBSCAN" + " on whole set...")
        db = cl.DBSCAN(eps = epsi, min_samples = mnpts, metric = "manhattan") #the algorithm
        clusters = db.fit_predict(disk_only) #the important returned vale
    #    centers = km.cluster_centers_.copy() #needed for variance
        
        #save them in case shit later screws up
        #save only if a good amount of clusters
        if clusters.cumsum()[-1] != len(clusters):
            fd = open("data/dbscans/apo_fe_db_" + str(mnpts) + "_" + str(epsi) + ".pkl", "wb")
            cp.dump(clusters, fd)
            fd.close()
        print("DBSCAN is done, we now inspect")
    
    
    def avg_mmetric(clusters, cluster, data):
        #function calculates the average mmetric between each two points in a cluster
        centre = np.zeros(len(data[0]))
        #calculate centre first
        for i in data[(clusters == cluster)]:
            centre += i
        centre = centre/len(data[(clusters == cluster)])
        distances = []
        for i in data[(clusters == cluster)]:
            distance = (np.absolute(centre - i)).mean()
            distances.append(distance)
        avg = sum(distances)/len(distances)
        return avg
        
    all_avg_mmetrics = np.array(list(avg_mmetric(clusters, i, disk_only) for i in range(-1, max(clusters)+1)))
    #noise_spread = avg_mmetric(clusters, -1, disk_only)
    sizes = np.array(list(sum(clusters == i) for i in range(-1, max(clusters)+1)))
    return sizes, all_avg_mmetrics, clusters

def plot_multi():
    params_1 = (8, 0.5)
    params_2 = (4, 0.5)
    params_3 = (2, 0.5)
    colours = ['r', 'g', 'b']
    params = [params_1, params_2, params_3]
    for param in params:
        x = run_dbscan(param[0], param[1])
        plt.semilogx(x[0], x[1], colours[0] + '.')
        colours = colours[1:]
        
    plt.title("DBSCAN CHANGE TITLE TO SUIT YOUR WHATEVER")
    plt.xlabel("Number in cluster")
    plt.ylabel("Average distance from cluster centre")
    legend = mpatches.Patch(color='white', label="(minpts, eps)")
    red_patch = mpatches.Patch(color='red', label=str(params[0]))
    green_patch = mpatches.Patch(color='green', label=str(params[1]))
    blue_patch = mpatches.Patch(color='blue', label=str(params[2]))
    plt.legend(handles=[legend, red_patch, green_patch, blue_patch], loc = 1)

params = (5, 0.27)
x = run_dbscan(params[0], params[1])
plt.semilogx(x[0], x[1], '.')

def chem_analysis(data_array):
    #find centre of them and calc distance between centre then between any two
    ss_centre = np.array([data_array.T[i].mean() for i in range(len(data_array[0]))])
    centre_dist = []
    for i in data_array:
        x = i - ss_centre
        y = np.absolute(x)
        z = sum(y)/len(ss_centre)
        centre_dist.append(z)
    centre_dist = np.array(centre_dist)
    
    pairwise = []
    count = 0
    for i in data_array:
        for j in data_array:
            if np.array_equal(i, j) == False:
                count += 1
                x = i - j
                y = np.absolute(x)
                z = sum(y)/len(ss_centre)
                pairwise.append(z)        
    pairwise = np.array(pairwise)
    return centre_dist, pairwise

#below is experimental and time consuming!!!
#print("open the newest shite apogee data")
#hdulist = fits.open('data/allStar-l30e.2.fits')
#tbdata = hdulist[1].data
#metadata_labels = ["RA", "DEC", "PMRA", "PMDEC", "PM_SRC", "GLON", "GLAT", "VHELIO_AVG", "FIELD", "APOGEE_ID"]
#metadata = np.vstack((tbdata.field(label) for label in metadata_labels)).T

