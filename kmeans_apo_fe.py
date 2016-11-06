#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:45:03 2016

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.io import ascii
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

#now run kmeans then take the densest clusters and mask the IDs!!!
#checking for the file and running if we don't have it
def run_kmeans(K):
    try:
        fd = open("data/apo_fe_km" + str(K) + ".pkl", "rb")
        clusters = cp.load(fd)
        fd.close()
        fd = open("data/apo_fe_km" + str(K) + "_centers.pkl", "rb")
        centers = cp.load(fd)
        fd.close()
        print("loaded clusters and centers")
    except:
        print("no clusters file so perform kmeans")
        print("running kmeans, k = " + str(K) + " on whole set...")
        
        km = cl.KMeans(n_clusters = K, random_state=42, n_init=32) #the algorithm
        clusters = km.fit_predict(disk_only) #the important returned vale
        centers = km.cluster_centers_.copy() #needed for variance
        
        #save them in case shit later screws up
        fd = open("data/apo_fe_km" + str(K) + ".pkl", "wb")
        cp.dump(clusters, fd)
        fd.close()
        fd = open("data/apo_fe_km" + str(K) + "_centers.pkl", "wb")
        cp.dump(centers, fd)
        fd.close()
        print("kmeans is done, we now compute density")
        
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
    
    all_avg_mmetrics = np.array(list(avg_mmetric(clusters, i, disk_only) for i in range(max(clusters)+1)))
    sizes = np.array(list(sum(clusters == i) for i in range(max(clusters)+1)))
    return sizes, all_avg_mmetrics, clusters

def plot_multi():
    Ks = [256, 512, 1024, 4096]
    colours = ['r', 'g', 'b', 'm']
    for k in Ks:
        x = run_kmeans(k)
        plt.semilogx(x[0], x[1], colours[0] + '.')
        colours = colours[1:]
        
    plt.title("Varying Kmeans Parameters")
    plt.xlabel("Number in cluster")
    plt.ylabel("Average distance from cluster centre")
    legend = mpatches.Patch(color='white', label='K')
    red_patch = mpatches.Patch(color='red', label=str(Ks[0]))
    green_patch = mpatches.Patch(color='green', label=str(Ks[1]))
    blue_patch = mpatches.Patch(color='blue', label=str(Ks[2]))
    magenta_patch = mpatches.Patch(color='magenta', label=str(Ks[3]))
    plt.legend(handles=[legend, red_patch, green_patch, blue_patch, magenta_patch])

K = 512
kmeans = run_kmeans(K)
clusters = kmeans[2]
#now compute variance Hogg style
#print("computing density...")
N, D = disk_only.shape
sizes = np.zeros(K).astype(int)
logdets = np.zeros(K)
#fehs = np.zeros(K)
for k in range(K):
    I = (clusters == k)
    sizes[k] = np.sum(I)
    subdata = disk_only[I,:] - (np.mean(disk_only[I], axis=0))[None, :]
    #i have commented some stuff out, including some of the smaller clusters
    #being returned, we will take them no matter what
#    if sizes[k] > (D + 1):
    variance = np.sum(subdata[:,:,None] * subdata[:,None,:], axis=0) / (sizes[k] - 1.)
    s, logdets[k] = np.linalg.slogdet(variance)
#    assert s > 0
#    else:
#        logdets[k] = -np.Inf
#    fehs[k] = np.mean(disk_only[I, 0])
densities = sizes * np.exp(-0.5 * logdets)
#
##plot the clusters for an idea
#plt.loglog(sizes, densities, '.')
#plt.xlabel("Cluster Size")
#plt.ylabel("Cluster Density")
#
##now how to get our favourite ones?
#keep = 40
#print("converging on top " + str(keep) + " results...")
#densest = []
#d = densities.copy()
#while len(densest) < 40:
#    densest.append(d.argmax())
#    d_mask = (d != d.max())
#    d = d[d_mask]
#
#
#densest_clusters = []
#for number in densest:
#    subset = (clusters == number)
#    add_cluster = disk_IDs[subset]
#    densest_clusters.append(add_cluster)
    
#alternatively, keep all
keep = "all"
densest_clusters = []
for number in range(clusters.max()):
    subset = (clusters == number)
    add_cluster = disk_IDs[subset]
    densest_clusters.append(add_cluster)


print("writing the " + str(keep) + " densest clusters to pickle")
fd = open("data/apo_fe_km512_" + str(keep) + ".pkl", "wb")
cp.dump(densest_clusters, fd)
fd.close()


