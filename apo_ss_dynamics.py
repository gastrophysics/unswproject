#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:05:19 2016

@author: john
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.actionAngle import estimateDeltaStaeckel
from galpy.actionAngle import actionAngleStaeckel
from galpy.potential import KeplerPotential
from galpy.util import bovy_conversion
from astropy import units
import pickle as cp
from astropy.io import ascii

#load up files with various candidates
all_canidates = []
for fn in ['apo_candidates_1']: # only one for now, 'apo_candidates_2', 'apo_candidates_3', 'apo_candidates_4']:  
    fd = open("data/" + fn + ".pkl", "rb")
    data = cp.load(fd)
    fd.close()
    for d in data:
        if d not in all_canidates:
            all_canidates.append(d)
print("candidates number: ", len(all_canidates[0]))

#grab the ages
solar_age = 4.6*(10**9) #try and grab an astropy constant here
#age_table = open('data/apo_ages.dat')
#age_and_mass = ascii.read(age_table.readlines())
##here we have
#ids_with_age = "col2"
#ln_mass = "col7"
#ln_age = "col8"
#ln_mass_e = "col15"
#ln_age_e = "col16"

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

#candidate orbital values
candidate_dist = []
candidate_ids = all_canidates[0].T[-1]
for ID in candidate_ids:
    for i in dist_table:
        if ID == i[0]:
            candidate_dist.append(i)
candidate_dist = dict(candidate_dist)      

#inspect the candidates if we want
#x = 1
#print("candidates for condition:", x)    
#x += 1
#for ID in candidate_ids:
#    for i in metadata:
#        if ID == i[-1]:
#            print(i)


#grab the UCACA4 values
print("grabbing ucac4 spatial data for ss candidates...")
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
ucac_dict[apo_id] = [ucac4_id, ra, ra_err, dec, dec_err, pmra, pmra_err, pmdec, pmdec_err]
"""

#get orbits for each of them with schoenrich             
print("initialising 3D orbits for solar siblings...")
orbits = []
for i in all_canidates[0]:
    ra = float(ucac_dict[i[-1]][1])
    dec = float(ucac_dict[i[-1]][3])
    dist = float(candidate_dist[i[-1]])
    pmRA = float(ucac_dict[i[-1]][5])
    pmDec = float(ucac_dict[i[-1]][7])
    Vlos = float(i[6])
    orbits.append(Orbit(vxvv=[ra,dec,dist,pmRA,pmDec,Vlos],radec=True,ro=8.,vo=220., solarmotion = "schoenrich"))

#so we have all wee need to calc an orbit
ts = np.linspace(0,-150,10000)
mwp= MWPotential2014

#sun's orbit for comparison:
sun = Orbit(vxvv=[0, 0, 0, 0, 0, 0],radec=True,ro=8.,vo=220., solarmotion = "schoenrich")
sun.integrate(ts,mwp,method='odeint')

mystuff = []
#plt.plot(sun.R(ts), sun.z(ts), 'r')
#colours = ['k', 'g', 'b', 'm', 'y']
for o in orbits:
    o.integrate(ts,mwp,method='odeint')
    if o.zmax() < 3*sun.zmax():
        mystuff.append(o)
#    plt.plot(o.x(ts), o.y(ts))
#    colours = colours[1:]


#distance portion
for o in mystuff:
    delta_x = o.x(ts) - sun.x(ts)
    delta_y = o.y(ts) - sun.y(ts)
    delta_z = o.z(ts) - sun.z(ts)
    delta = (delta_x**2 + delta_y**2 + delta_z**2)**0.5
    plt.semilogy(sun.time(ts), delta*1000)



#action angles portion:
#actionanglesAdiabatic for MWpotential2014
#if we want it for comparison:
#aAA_sun = actionAngleAdiabatic(pot=MWPotential2014)
#js_aaa_sun = aAA_sun(sun.R(ts),sun.vR(ts),sun.vT(ts),sun.z(ts),sun.vz(ts))
#

#convert into natural shit
ro = 8
vo = 220
delta_sun = estimateDeltaStaeckel(MWPotential2014,(sun.R(ts))/ro, (sun.z(ts))/vo)
aAS_sun = actionAngleStaeckel(pot=MWPotential2014,delta = delta_sun)
js_sun = aAS_sun((sun.R())/ro,(sun.vR())/vo,(sun.vT())/vo,(sun.z())/ro,(sun.vz())/vo)
mean_js = np.array([js_sun[0].mean(), js_sun[1].mean(), js_sun[2].mean()])

print("going for the big fella")
for o in orbits:
    try:
        #quick method: only first instance
        delta_o = estimateDeltaStaeckel(MWPotential2014,(o.R(ts))/ro,(o.z(ts))/ro)
    #    plt.plot(js[0] - js_sun[0], ts)
        aAS = actionAngleStaeckel(pot=MWPotential2014,delta = delta_o)
        js_o = aAS((o.R())/ro,(o.vR())/vo,(o.vT())/vo,(o.z())/ro,(o.vz())/vo)
        mean_o_js = np.array([js_o[0].mean(), js_o[1].mean(), js_o[2].mean()])
        mean_js = np.vstack([mean_js, mean_o_js])
    except:
        #problem with this orbit (probs unbound)
        #let it just be big
        mean_o_js = np.array([1000*js_sun[0].mean(), 1000*js_sun[1].mean(), 1000*js_sun[2].mean()])
        mean_js = np.vstack([mean_js, mean_o_js])

fd = open("data/mean_js_ss_t100.pkl", "wb")
cp.dump(mean_js ,fd)
fd.close()

percentage_diffs = np.absolute((mean_js[1:] - mean_js[0]))/mean_js[0]

#as per the saved figure
plt.semilogy(percentage_diffs, 'o')
plt.title("Difference of Actions to Solar Orbit")
plt.axhline(5, linestyle = 'dashed', color = 'k')
plt.ylabel("Log Difference")
plt.xlabel("APOGEE ID")
plt.xlim(-1,18)
x_labels = range(1, 19)
plt.xticks(x_labels, all_canidates[0].T[-1], rotation = -20)
