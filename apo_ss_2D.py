#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:11:44 2016

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
from math import tan, cos
from astropy.io import fits
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.actionAngle import estimateDeltaStaeckel
from galpy.actionAngle import actionAngleStaeckel
from galpy.potential import KeplerPotential, SteadyLogSpiralPotential
from galpy.util import bovy_conversion
from astropy import units
import pickle as cp
from astropy.io import ascii

#load up files with various candidates
try:
    fd = open("data/2D_ss_ucac4_pre.pkl", "rb")
    orbits_2D = cp.load(fd)
    fd.close()
    print("orbits_2D loaded from file")
except:
    print("no file exists, bulding orbits from scratch...")
    all_candidates = []
    for fn in ['apo_candidates_1']: # only one for now, 'apo_candidates_2', 'apo_candidates_3', 'apo_candidates_4']:  
        fd = open("data/" + fn + ".pkl", "rb")
        data = cp.load(fd)
        fd.close()
        for d in data:
            if d not in all_candidates:
                all_candidates.append(d)
    print("candidates number: ", len(all_candidates[0]))
    
    #grab the ages
    solar_age = 4.6*(10**9) #try and grab an astropy constant here
    age_table = open('data/apo_ages.dat')
    age_and_mass = ascii.read(age_table.readlines())
    #here we have
    ids_with_age = "col2"
    ln_mass = "col7"
    ln_age = "col8"
    ln_mass_e = "col15"
    ln_age_e = "col16"
    
    #inspect them if we want
    #for ID in candidate_ids:
    #    for i in age_and_mass:
    #        if ID == i[1]:
    #            print(ID, ":", i[8])
    
    
    
    #grab the distances for these candidates
    hdudist = fits.open('data/allStar+-v603.150209.fits')
    distances = hdudist[1].data.field('DISO').T[1].T
    ap_id = hdudist[1].data.field('APOGEE_ID')
    dist_table = np.vstack([ap_id, distances]).T
    
    candidate_dist = []
    candidate_ids = all_candidates[0].T[-1]
    for ID in candidate_ids:
        for i in dist_table:
            if ID == i[0]:
                candidate_dist.append(i)
    candidate_dist = dict(candidate_dist)      
    
    #grab the UCACA4 values
    ucac = np.loadtxt('data/ss_ucac4.csv', dtype = str, delimiter=',')
    #clean it up
    for i in range(len(ucac)):
        for e in range(len(ucac[i])):
            ucac[i][e] = ucac[i][e][2:-1]
    #make a dcit with apo_id as key and ucac values as a list
    ucac_dict = []
    for i in ucac:
        ucac_dict.append([i[0], i[1:]])
    ucac_dict = dict(ucac_dict)
    
    """
    ucac4 values work as follows with apo id as the key
    ucac_dict = [ucac4_id, ra, ra_err, dec, dec_err, pmra, pmra_err, pmdec, pmdec_err]
    """
    #inspect the candidates if we want
    #x = 1
    #print("candidates for condition:", x)    
    #x += 1
    #for ID in candidate_ids:
    #    for i in metadata:
    #        if ID == i[-1]:
    #            print(i)
    
    print("initialising 3D orbits for solar siblings...")
    #get orbits for each of them with schoenrich             
    orbits = []
    for i in all_candidates[0]:
        ra = float(ucac_dict[i[-1]][1])
        dec = float(ucac_dict[i[-1]][3])
        dist = float(candidate_dist[i[-1]])
        pmRA = float(ucac_dict[i[-1]][5])
        pmDec = float(ucac_dict[i[-1]][7])
        Vlos = float(i[6])
        orbits.append(Orbit(vxvv=[ra,dec,dist,pmRA,pmDec,Vlos],radec=True,ro=8.,vo=220., solarmotion = "schoenrich"))
    
    #3D
    #so we have all wee need to calc an orbit back in time
    ts = np.linspace(0,-150,10000)
    mwp = MWPotential2014
    
    #sun's orbit for comparison:
    sun = Orbit(vxvv=[0, 0, 0, 0, 0, 0],radec=True,ro=8.,vo=220., solarmotion = "schoenrich")
    sun.integrate(ts,mwp,method='odeint')
    
    print("Selecting orbits with low z")
    #choose only the ones with low z
    orbits_2D = []
    for o in orbits:
        o.integrate(ts,mwp,method='odeint')
        if o.zmax() < 5*sun.zmax():
            o.plot()
            orbits_2D.append(o)
            
    print("Candidates reduced to " + str(len(orbits_2D)))
            
    #2D
    sun_2D = sun.toPlanar()
    orbits_2D.append(sun_2D)
    
    fd = open("data/2D_ss_ucac4_pre.pkl", "wb")
    cp.dump(orbits_2D ,fd)
    fd.close()

print("Preparing data for 2D analysis")
ts = np.linspace(0,-150,10000)
mwp = MWPotential2014
print("Flattening orbits and potential")
mwp_2D = [i.toPlanar() for i in mwp]    
sun_2D = orbits_2D[-1]
sun_2D.integrate(ts,mwp_2D,method='odeint')
orbits_2D = orbits_2D[:-1]

#spiral parameters and defaults
sp_m = 4    #4
sp_spv = 20 #20
sp_i = -15*(3.1415/180) #-12
sp_x_0 = -150*(3.1415/180)   #-120
sp_a = -sp_m/tan(sp_i)
sp_gamma = sp_x_0/sp_m
sp_fr_0 = 0.05  #0.05
ro = 8
vo = 220
sp_A = -sp_fr_0 #*((ro*vo)**2) #not sure on this part but adding these makes it too big so leave them


sp_component = SteadyLogSpiralPotential(amp = 1, omegas = sp_spv, A = sp_A, alpha = sp_a, gamma = sp_gamma)
mwp_2D.append(sp_component) # not yet

print("finally calculating distance from sun...")
for o in orbits_2D:
    o_2D = o.toPlanar()
    o_2D.integrate(ts,mwp_2D,method='odeint')
    delta_x = o_2D.x(ts) - sun_2D.x(ts)
    delta_y = o_2D.y(ts) - sun_2D.y(ts)
    delta = (delta_x**2 + delta_y**2)**0.5
    plt.semilogy(sun_2D.time(ts), delta*1000)

plt.title("Solar Sibling Proximity")
plt.xlabel("Time (Gyr)")
plt.ylabel("Distance (pc)")
#plt.axhline(y = 100, color='r')

    
