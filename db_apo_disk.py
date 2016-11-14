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

def run_dbscan(mnpts, epsi, chem_data = disk_only ,stacking = False):
    #to avoid runnning the same scan twice
    if stacking == False:
        try:
            fd = open("data/dbscans/apo_fe_db_" + str(mnpts) + "_" + str(epsi) + ".pkl", "rb")
            clusters = cp.load(fd)
            fd.close()
            print("loaded clusters from file")
        except:
            print("no clusters file so perform DBSCAN")
            print("running DBSCAN" + " on whole set...")
            db = cl.DBSCAN(eps = epsi, min_samples = mnpts, metric = "manhattan") #the algorithm
            clusters = db.fit_predict(chem_data) #the important returned vale
        #    centers = km.cluster_centers_.copy() #needed for variance
            
            #save them in case shit later screws up
            #save only if a good amount of clusters
            if clusters.cumsum()[-1] != len(clusters):
                fd = open("data/dbscans/apo_fe_db_" + str(mnpts) + "_" + str(epsi) + ".pkl", "wb")
                cp.dump(clusters, fd)
                fd.close()
            print("DBSCAN is done, we now inspect")
#    the above only applies when we want the scan run on thw whole set for the first time
#    if we are stacking them then we want to perform it no matter what
    if stacking == True:
        db = cl.DBSCAN(eps = epsi, min_samples = mnpts, metric = "manhattan") #the algorithm
        clusters = db.fit_predict(chem_data) #the important returned vale
        
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
        
    all_avg_mmetrics = np.array(list(avg_mmetric(clusters, i, chem_data) for i in range(-1, max(clusters)+1)))
    #noise_spread = avg_mmetric(clusters, -1, chem_data)
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

#params = (5, 0.27)
#x = run_dbscan(params[0], params[1])
#plt.semilogx(x[0], x[1], '.')

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


def dbscan_stacking(stacks, stuff):
#    colours = ['r', 'g', 'b', 'm', 'c', 'k']
    params = [(5, 0.27), (5, 0.28), (5, 0.29), (5, 0.295), (5, 0.297), (5, 0.34)]
    params = [(5, eps) for eps in np.linspace(0.27, 0.44, 35)] #to show progression
    def name_file(iteration):
        #naming the file is difficult since we stack so many in some different order 
        #each time. We want to pick up from where we left off so the name
        #depends on what iteration we are on and the params list values
        name = ""
        for it in range(iteration+1):
            name += "_"
            name += "iter" + str(it) + ":"
            name += str(params[it][0])
            name += "_"
            name += str(params[it][1])
        return name
    report = []
    cluster_numbers = []
    k = 1 #this is a naming thing
    for i in range(stacks):
        if i > 15:
            name = "special" + str(k) #placeholder, name too long
            k += 1
        else:
            name = name_file(i)
        try:
            fd = open("data/dbscans/apo_fe_db" + name + ".pkl", "rb")
            current_stack = cp.load(fd)
            fd.close()
            x = current_stack[1]
            stuff = current_stack[0]
            if x[-1].max() <= 1000: #currently not being used properly
                rep = []
                for j in range(x[-1].max()):
                    if stuff[(x[-1] == j)].shape[0] >= 5: #only compute if a good cluster, sometimes it returns low???
                    #calculate pairwise info for each cluster, ive left three here so you can change it around
                        pwmean = chem_analysis(stuff[(x[-1] == j)])[1].max()    
                        pwmax = chem_analysis(stuff[(x[-1] == j)])[1].max()  
                        pwmin = chem_analysis(stuff[(x[-1] == j)])[1].max()  
                        pw = [pwmin, pwmean, pwmax]
                        rep.append(pw)
                #create summary of pairwise info for clusters on this scan
                rep = np.vstack(np.array(pw) for pw in rep)
                iteration_min = rep.T[0].min()
                iteration_max = rep.T[2].max()
                iteration_mean = rep.T[1].mean()
                iteration_summary = [iteration_min, iteration_mean, iteration_max]
                report.append(iteration_summary)
            plt.semilogx(x[0], x[1], '.') #change '.' to colours[i] + '.' when using normal
            stuff = stuff[(x[-1] == -1)]
            cluster_numbers.append(x[-1].max())
            print("shorcutting iteration " + str(i))
        except:
            print("runnning stack " + str(i))
            x = run_dbscan(params[i][0], params[i][1], chem_data = stuff, stacking = True)
            plt.semilogx(x[0], x[1], '.') #change '.' to colours[i] + '.' when using normal
            #so that we don't compute pairwise on noise or runaway
            if x[-1].max() <= 1000: #currently not being used properly
                rep = []
                for j in range(x[-1].max()):
                    if stuff[(x[-1] == j)].shape[0] >= 5: #only compute if a good cluster, sometimes it returns low???
                    #calculate pairwise info for each cluster, ive left three here so you can change it around
                        pwmean = chem_analysis(stuff[(x[-1] == j)])[1].max()    
                        pwmax = chem_analysis(stuff[(x[-1] == j)])[1].max()  
                        pwmin = chem_analysis(stuff[(x[-1] == j)])[1].max()  
                        pw = [pwmin, pwmean, pwmax]
                        rep.append(pw)
                #create summary of pairwise info for clusters on this scan
                rep = np.vstack(np.array(pw) for pw in rep)
                iteration_min = rep.T[0].mean()
                iteration_max = rep.T[2].mean()
                iteration_mean = rep.T[1].mean()
                iteration_summary = [iteration_min, iteration_mean, iteration_max]
                report.append(iteration_summary)          
            current_stack = (stuff, x)
            stuff = stuff[(x[-1] == -1)]
            cluster_numbers.append(x[-1].max())
            fd = open("data/dbscans/apo_fe_db" + name + ".pkl", "wb")
            cp.dump(current_stack, fd)
            fd.close()

    plt.title("DBSCAN CHANGE TITLE TO SUIT YOUR WHATEVER")
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("Number in cluster")
    plt.ylabel("Average distance from cluster centre")
#    legend = mpatches.Patch(color='white', label="(minpts, eps)")
#    red_patch = mpatches.Patch(color='red', label=str(params[0]))
#    green_patch = mpatches.Patch(color='green', label=str(params[1]))
#    blue_patch = mpatches.Patch(color='blue', label=str(params[2]))
#    mag_patch = mpatches.Patch(color='magenta', label=str(params[3]))
#    cy_patch = mpatches.Patch(color='cyan', label=str(params[4]))
#    black_patch = mpatches.Patch(color='black', label=str(params[5]))
#    plt.legend(handles=[legend, red_patch, green_patch, blue_patch, mag_patch, cy_patch, black_patch], loc = 2)
    report = np.vstack(np.array(summary) for summary in report)
    return report, cluster_numbers
#below is experimental and time consuming!!!
#print("open the newest shite apogee data")
#hdulist = fits.open('data/allStar-l30e.2.fits')
#tbdata = hdulist[1].data
#metadata_labels = ["RA", "DEC", "PMRA", "PMDEC", "PM_SRC", "GLON", "GLAT", "VHELIO_AVG", "FIELD", "APOGEE_ID"]
#metadata = np.vstack((tbdata.field(label) for label in metadata_labels)).T

