#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:52:22 2016

@author: john
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.potential import KeplerPotential
from galpy.actionAngle import estimateDeltaStaeckel
from galpy.actionAngle import actionAngleStaeckel
from galpy.util import bovy_conversion
from astropy import units
import pickle as cp
from astropy.io import ascii

# lets say we have some files with the 10 densest clusters and load them up
# they will have a structure like so:
# pickled_file_1 = [[IDs_1], [IDs_2], [IDs_3]...[IDs_10]]
# pickled_file_2 = [[IDs_1], [IDs_2], [IDs_3]...[IDs_10]]
# etc.
files = ["apo_fe_km512_all"] #empty for now, but files = [pickled_files_1, pickled_files_2, etc.]
selection_criteria_dict = {} #some dict containing the file name and 
                                # a desciption of it for use on graphs later

all_data = []
for file in files:
    fd = open("data/" + file + ".pkl", "rb")
    data = cp.load(fd)
    fd.close()
    all_data.append(data)

#now, all_data = [data_1, data_2, etc] data = [[IDs_1], [IDs_2], [IDs_3]...[IDs_10]]
def load_data():
    #this function grabs any data we may need, the file has been built and saved
    #leave this part alone unless you know what you're doing
    print("loading up chem, spatial and distances")
    try:
        fd = open("data/APO_3sources.pkl", "rb")
        print("fast load from saved data...")
        x = cp.load(fd)
        fd.close()
        
    except:
        print("slow load from 3 data tables...")
        #load up chemical data from Sarah's subset
        fe_data_source = open('data/stars-clusters.dat')
        c = fe_data_source.readlines()
        abundances = ascii.read(c)
        labels_Fe = ["APOGEE_ID", "FE_H",
              "AL_FE", "CA_FE", "C_FE", "K_FE",  "MG_FE", "MN_FE", "NA_FE",
              "NI_FE", "N_FE",  "O_FE", "SI_FE", "S_FE",  "TI_FE", "V_FE"]
        chem_data = np.vstack((abundances.field(label) for label in labels_Fe)).T
    
        #grab the distances for everything from Sarah distance file
        hdudist = fits.open('data/allStar+-v603.150209.fits')
        distances = hdudist[1].data.field('DISO').T[1].T
        ap_id = hdudist[1].data.field('APOGEE_ID')
        dist_table = np.vstack([ap_id, distances]).T
    
        #grab spatial data from Hogg file
        hdulist = fits.open('data/results-unregularized-matched.fits')
        meta_source = hdulist[1].data
        #could include this data at some point
    #    labels_H = = ["FE_H",
    #               "AL_H", "CA_H", "C_H", "K_H",  "MG_H", "MN_H", "NA_H",
    #               "NI_H", "N_H",  "O_H", "SI_H", "S_H",  "TI_H", "V_H"]
        spatial_labels = ["RA", "DEC", "PMRA", "PMDEC", "VHELIO_AVG", "TEFF_ASPCAP", "LOGG_ASPCAP", "FIELD", "APOGEE_ID"]
        spatial_data = np.vstack((meta_source.field(label) for label in spatial_labels)).T
        print("saving the file for later use")
        fd = open("data/APO_3sources.pkl", "wb")
        #now we have 3 data thingos
        x = [chem_data, spatial_data, dist_table]
        cp.dump(x, fd)
        fd.close()
    return x
    
def retrieve_data(IDs_list):
    data_triplet = load_data()
    chem_data = data_triplet[0].copy()
    spatial_data = data_triplet[1].copy()
    dist_table = data_triplet[2].copy()
    #cut down data above to a [chem, spatial, dict{dist}] for just the IDs
    #initialize the mask as all false
    chem_mask = (chem_data.T[0] == 'placeholder')
    spatial_mask = (spatial_data.T[-1] == 'placeholder') #-1 in the spatial
    dist_mask = (dist_table.T[0] == 'placeholder')
    
    for ID in IDs_list:
        chem_mask += (chem_data.T[0] == ID)
        spatial_mask += (spatial_data.T[-1] == ID)
        dist_mask += (dist_table.T[0] == ID)
        
    chem_data = chem_data[chem_mask]
    spatial_data = spatial_data[spatial_mask]
    dist_table = dist_table[dist_mask] #can turn into dict if we want
    
    #the triplet we return is what we mean by "cluster"
    #it contains all the details of the given cluster    
    return [chem_data, spatial_data, dist_table]

def cspace_information(cluster):
    #function calculates the average mmetric between each two points in a cluster
    centre = np.zeros(len(cluster[0][0])-1)
    #calculate centre first
    for i in cluster[0]:
        centre += i[1:].astype(float)
    centre = centre/(len(cluster[0][0])-1)
    distances = []
    for i in cluster[0]:
        distance = (np.absolute(centre - i[1:].astype(float))).mean()
        distances.append(distance)
    avg = sum(distances)/len(distances)    
    return avg

def emp_density(cluster):
    #print("computing density...")
    data = np.delete(cluster[0], 0, 1)
    data = data.astype(float)
    N, D = data.shape
    subdata = data - (np.mean(data, axis=0))[None, :]
    variance = np.sum(subdata[:,:,None] * subdata[:,None,:], axis=0) / (N - 1.)
    s, logdets = np.linalg.slogdet(variance)
    density = N * np.exp(-0.5 * logdets)    
    return density, N
    
def cluster_dynamics(cluster):
    orbits = []
    dist_dict = dict(cluster[2])
    for i in cluster[1]:
        ra = float(i[0])
        dec = float(i[1])
        dist = float(dist_dict[i[-1]])
        pmRA = float(i[2])
        pmDec = float(i[3])
        Vlos = float(i[4])
        #only attach where proper motions are available
        if dist != 0.0:
            if pmRA != 0.0:
                if pmDec != 0.0:        
                    orbits.append(Orbit(vxvv=[ra,dec,dist,pmRA,pmDec,Vlos],radec=True,ro=8.,vo=220., solarmotion = "schoenrich"))
    ts = np.linspace(0,100,10000)
    mwp= MWPotential2014
    print(len(orbits))
    #action angles yay!
    #the following gets confusing because we speak of "means" twice
    #we want the mean J over each individual orbit, but to measure the spread in the cluster
    #we compare to the mean of each J
    ro = 8
    vo = 220
    mean_js = np.array([])
    for o in orbits:
        try:
            o.integrate(ts,mwp,method='odeint')
            delta_o = estimateDeltaStaeckel(MWPotential2014,(o.R())/ro,(o.z())/vo)
            aAS = actionAngleStaeckel(pot=MWPotential2014,delta = delta_o)
            js_o = aAS((o.R())/ro,(o.vR())/vo,(o.vT())/vo,(o.z())/ro,(o.vz())/vo)
            #the next line isn't really needed unless js is actually a vector, for t = 0 it is not
            mean_o_js = np.array([js_o[0].mean(), js_o[1].mean(), js_o[2].mean()])
            if mean_js.shape == (0,):
                mean_js = mean_o_js
            else:
                mean_js = np.vstack([mean_js, mean_o_js])
        except:
            print("found unbound, discarded")
            #in this case the orbit is unbound -- this fucks up the whole thing I know
            pass
    print(mean_js.shape)
    #so for this cluster we have a 3 x cluster length array
    cluster_mean_js = np.array([mean_js.T[0].mean(), mean_js.T[1].mean(), mean_js.T[2].mean()])
    #average_difference
    percentage_diffs = np.absolute((mean_js - cluster_mean_js))/cluster_mean_js
    dynamics_summary = np.array([percentage_diffs.T[0].mean(), percentage_diffs.T[1].mean(), percentage_diffs.T[2].mean()])
    return dynamics_summary

def full_analysis(clusters):
    cl = retrieve_data(clusters[0])
    chem_spread = cspace_information(cl)
    action_spread = cluster_dynamics(cl)
    dens = emp_density(cl)
    for cluster in clusters[1:]:
        cl = retrieve_data(cluster)
        chem_spread = np.hstack((chem_spread, cspace_information(cl)))
        action_spread = np.vstack((action_spread, cluster_dynamics(cl)))
        dens = np.vstack((dens, emp_density(cl)))
    return chem_spread, action_spread, dens
    
def density_check(clusters):
    cl = retrieve_data(clusters[0])
    dens = emp_density(cl)
    for cluster in clusters[1:]:
        cl = retrieve_data(cluster)
        dens = np.vstack((dens, emp_density(cl)))
    plt.semilogy(dens.T[1], dens.T[0], '.')

    
        
        