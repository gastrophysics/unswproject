#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:38:22 2016

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
from math import tan, cos
from astropy.io import fits
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic 
from galpy.actionAngle import estimateDeltaStaeckel, estimateBIsochrone
from galpy.actionAngle import actionAngleStaeckel, actionAngleIsochroneApprox
from galpy.potential import KeplerPotential, SteadyLogSpiralPotential
from galpy.util import bovy_conversion
from astropy import units
import pickle as cp
from astropy.io import ascii


#change the file name and test it out
def ucac_pms():
    ucac = np.loadtxt('data/UCAC.results', dtype = str)
    #clean it up
    for i in range(len(ucac)):
        for e in range(len(ucac[i])):
            ucac[i][e] = ucac[i][e][2:-1]
        #make a dcit with apo_id as key and ucac values as a list
    ucac_dict = []
    for i in ucac:
        ucac_dict.append([i[0], i[1:]])
    ucac_dict = dict(ucac_dict)
    return ucac_dict


#sort into clusters
def sorted_clusters():
    fd = open("data/dbscans/apo_fe_db_5_0.27.pkl", "rb")
    cluster_number = cp.load(fd)
    fd.close()
    
    thick = open('data/stars-thickdisk-2.dat')
    thin = open('data/stars-thindisk-2.dat')
    components = [thick, thin]
    comp_abundances = []
    for component in components:
        c = component.readlines()
        abundances = ascii.read(c)
        comp_abundances.append(abundances)
    #reduce to just the abundances and IDs
    meta_labels = ["APOGEE_ID"]
    IDs = []
    for component in comp_abundances:
        ID = np.vstack((component.field(label) for label in meta_labels)).T
        IDs.append(ID)
    
    disk_IDs = np.vstack((IDs[0], IDs[1]))
    disk_IDs = disk_IDs.T[0]
    
    clusters = []     
    all_ids = disk_IDs
    for number in range(cluster_number.max() + 1):
        cluster = all_ids[(cluster_number == number)]
        clusters.append(cluster)
    return clusters
    #number = 0
    #for cluster in clusters:
    #    print("cluster: " + str(number))
    #    print(cluster)
    #    number += 1

def grab_other_spatials(cls):
    #have to grab distance, ra, dec, vlos
    hdulist = fits.open('data/results-unregularized-matched.fits')
    tbdata = hdulist[1].data
    metadata_labels = ["RA", "DEC", "PMRA", "PMDEC", "GLON", "GLAT", "VHELIO_AVG", "TEFF_ASPCAP", "LOGG_ASPCAP", "FIELD", "APOGEE_ID"]
    metadata = np.vstack((tbdata.field(label) for label in metadata_labels)).T
    position_dict = []
    for cl in cls:
        other_spatial = []
        for star in cl:
            other_spatial.append(metadata[metadata.T[-1] == star])            
        for spatial in other_spatial:
            spatial = spatial[0]
            position_dict.append([spatial[-1],[spatial[0], spatial[1], spatial[6]]])
    position_dict = dict(position_dict)
    
    #grab the distances for apogee
    hdudist = fits.open('data/allStar+-v603.150209.fits')
    distances = hdudist[1].data.field('DISO').T[1].T
    ap_id = hdudist[1].data.field('APOGEE_ID')
    dist_table = np.vstack([ap_id, distances]).T
    
    #candidate orbital values
    distance_dict = []
    for cl in cls:
        for star in cl:
            for i in dist_table:
                if star == i[0]:
                    distance_dict.append(i)
    distance_dict = dict(distance_dict)   
    return position_dict, distance_dict
        
    
def action_analysis(cls):
    try:
        fd = open("data/db_9cl_orbits_good.pkl", "rb")
        all_orbits = cp.load(fd)
        fd.close()
    except:
        ucac_dict = ucac_pms()
        position_dict = grab_other_spatials(cls)[0]
        distance_dict = grab_other_spatials(cls)[1]
        all_orbits = []
        for cl in cls:
            cluster_orbits = []
            for star in cl:
                try:
                    ra = float(position_dict[star][0])
                    dec = float(position_dict[star][1])
                    dist = float(distance_dict[star])
                    pmRA = float(ucac_dict[star][0])
                    pmDec = float(ucac_dict[star][2])
                    Vlos = float(position_dict[star][2])
                    #only attach where proper motions are available
                    if ucac_dict[star][-1] == '1':
                        cluster_orbits.append(Orbit(vxvv=[ra,dec,dist,pmRA,pmDec,Vlos],radec=True,ro=8.,vo=220., solarmotion = "schoenrich"))
                except:
                    print("bad egg, spitting out")
                    continue
                #only attach where proper motions are available
            all_orbits.append(cluster_orbits)
            print("ticking: ", len(all_orbits))
        fd = open("data/db_9cl_orbits_good.pkl", "wb")
        cp.dump(all_orbits, fd)
        fd.close()
    
    ts = np.linspace(0,100,10000)
    mwp= MWPotential2014
    #action angles yay!
    #the following gets confusing because we speak of "means" twice
    #we want the mean J over each individual orbit, but to measure the spread in the cluster
    #we compare to the mean of each J
    ro = 8
    vo = 220
    all_actions = []
    fig = 1
    for cluster_orbits in all_orbits:
        all_js = np.array([])
        for o in cluster_orbits:
            o.integrate(ts,mwp,method='odeint')
#            plt.plot(o.R(ts), o.z(ts))
            try:
                delta_o = estimateDeltaStaeckel(MWPotential2014,(o.R(ts))/ro,(o.z(ts))/ro)
                aAS = actionAngleStaeckel(pot=MWPotential2014,delta = delta_o)
                js_o = aAS((o.R())/ro,(o.vR())/vo,(o.vT())/vo,(o.z())/ro,(o.vz())/vo)
#                delta_b = estimateBIsochrone(MWPotential2014, (o.R(ts))/ro,(o.z(ts))/ro)
#                delta_b = delta_b[1]
#                aAIA = actionAngleIsochroneApprox(pot=MWPotential2014,b = delta_b)
#                js_o = aAIA((o.R())/ro,(o.vR())/vo,(o.vT())/vo,(o.z())/ro,(o.vz())/vo, nonaxi = True)
#                print(js_o)
                if all_js.shape == (0,):
                    all_js = js_o
                else:
                    all_js = np.vstack([all_js, js_o])
            except:
                print("found unbound, discarded")    
        print(np.array(all_js).shape)
        all_actions.append(all_js)
#        plt.xlabel("R distance (kpc)")
#        plt.ylabel("z distance (kpc)")
#        plt.xlim(0, 20)
#        plt.ylim(-5, 5)
#        plt.title("Cluster " + str(fig)+ " Rz" + " Orbits")
#        fn = open("dbcl_" + str(fig) + "_Rz" + ".png", "wb")
#        plt.savefig("dbcl_" + str(fig) + "_Rz" + ".png")
#        fn.close()
#        plt.close()
        fig += 1
    return all_actions

def let_loose():
    cls = sorted_clusters()
    actions = action_analysis(cls)
    return actions
