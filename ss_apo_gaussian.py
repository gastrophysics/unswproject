#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:19:29 2016

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

try:
    fd = open("data/arms_apogee_data.pkl", "rb")
    arms_apogee_data = cp.load(fd)
    fd.close()
    print("ids, vlos, and ucac data loaded from file")
except:
    candidate_ids = ['2M06150356-0021091', '2M04221969+4439373', '2M06434070+0051170']
    vlos_dict = [[candidate_ids[0], 11.864100456237793], [candidate_ids[1], 46.98500061035156], [candidate_ids[2], 26.171199798583984]]
    vlos_dict = dict(vlos_dict)
    #grab the ages
    #solar_age = 4.6*(10**9) #try and grab an astropy constant here
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
    
    candidate_dist = []
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
    
    arms_apogee_data = [candidate_ids, vlos_dict, ucac_dict, candidate_dist]
    
    fd = open("data/arms_apogee_data.pkl", "wb")
    cp.dump(arms_apogee_data, fd)
    fd.close()
    print("saved some shit")

    
def generate_rand(measurement, error, measurement_type = None):
    #the type is the important part for converting the measurement
    if measurement_type == 'mas':
        error = error/3600000
    value = np.random.normal(loc = measurement, scale = error)
    return value
    
#arm_apogee_data = [candidate_ids, vnolos_dict, ucac_dict]
candidate_ids = arms_apogee_data[0]
vlos_dict = arms_apogee_data[1]
ucac_dict = arms_apogee_data[2]
candidate_dist = arms_apogee_data[3]

"""
ucac4 values work as follows with apo id as the key
ucac_dict = [ucac4_id, ra, ra_err, dec, dec_err, pmra, pmra_err, pmdec, pmdec_err]
"""
print("initialising 3D orbits for solar siblings...")
#get orbits for each of them with schoenrich             
def initialise_orbit(ID):
    ra_mean = float(ucac_dict[ID][1])
    ra_err = float(ucac_dict[ID][2])
    dec_mean = float(ucac_dict[ID][3])
    dec_err = float(ucac_dict[ID][4])
    pmRA_mean = float(ucac_dict[ID][5])
    pmRA_err = float(ucac_dict[ID][6])
    pmDec_mean = float(ucac_dict[ID][7])
    pmDec_err = float(ucac_dict[ID][8])
    
    #calc a random value from gaussian
    ra = generate_rand(ra_mean, ra_err, measurement_type = 'mas')
    dec = generate_rand(dec_mean, dec_err, measurement_type = 'mas')
    pmRA = generate_rand(pmRA_mean, pmRA_err)
    pmDec = generate_rand(pmDec_mean, pmDec_err)
    #no error for distance in this set
    dist = float(candidate_dist[ID])
    
    #VLOS is reliable and stays as is
    Vlos = float(vlos_dict[ID])
    orbit = Orbit(vxvv=[ra,dec,dist,pmRA,pmDec,Vlos],radec=True,ro=8.,vo=220., solarmotion = "schoenrich")
    return orbit
    
#sun's orbit for comparison:
print("Preparing data for 2D analysis")
ts = np.linspace(0,-150,10000)
mwp = MWPotential2014
sun = Orbit(vxvv=[0, 0, 0, 0, 0, 0],radec=True,ro=8.,vo=220., solarmotion = "schoenrich")
sun.integrate(ts,mwp,method='odeint')
print("Flattening orbits and potential")
mwp_2D = [i.toPlanar() for i in mwp]   
sun_2D = sun.toPlanar()
sun_2D.integrate(ts,mwp_2D,method='odeint')

def add_spiral(arms):
    #spiral parameters and defaults then generate a value for it
    sp_m = arms    #4
    sp_spv = (22.5, 7.5) #20 15-30
    sp_spv = generate_rand(sp_spv[0], sp_spv[1])
    
    if sp_m == 4:
        sp_i = (-12*(3.1415/180), 2*(3.1415/180)) #12, err = 2      6, err = 1 (2 arms)
        sp_i = generate_rand(sp_i[0], sp_i[1])
    if sp_m == 2:
        sp_i = (-6*(3.1415/180), 1*(3.1415/180)) #12, err = 2      6, err = 1 (2 arms)
        sp_i = generate_rand(sp_i[0], sp_i[1])
    
    sp_x_0 = (-120*(3.1415/180), 10*(3.1415/180))   #-120 err = 10
    sp_x_0 = generate_rand(sp_x_0[0], sp_x_0[1])
    
    sp_a = -sp_m/tan(sp_i)
    sp_gamma = sp_x_0/sp_m
    sp_fr_0 = (0.05, 0.01)
    sp_fr_0 = generate_rand(sp_fr_0[0], sp_fr_0[1])
#    ro = 8
#    vo = 220
    sp_A = -sp_fr_0 #*((ro*vo)**2) #not sure on this part but adding these makes it too big so leave them
    sp_component = SteadyLogSpiralPotential(amp = 1, omegas = sp_spv, A = sp_A, alpha = sp_a, gamma = sp_gamma)
    mwp_2D.append(sp_component) # not yet

ID = candidate_ids[0]
print("finally calculating distance from sun for 100 iterations " + ID)
counts = []
for ID in candidate_ids:
    count = 0
    for i in range(1000):
        add_spiral(4)
        o = initialise_orbit(ID)
        o_2D = o.toPlanar()
        o_2D.integrate(ts,mwp_2D,method='odeint')
        delta_x = o_2D.x(ts) - sun_2D.x(ts)
        delta_y = o_2D.y(ts) - sun_2D.y(ts)
        delta = (delta_x**2 + delta_y**2)**0.5
        delta = delta*1000
    #    plt.semilogy(sun_2D.time(ts), delta*1000)
        delta_before_3gyr = delta[(sun_2D.time(ts) <= -3)]
        if delta_before_3gyr.min() <= 100:
            count += 1
    counts.append(count)
print(counts)
    
        
        
#plt.title("Solar Sibling Proximity")
#plt.xlabel("Time (Gyr)")
#plt.ylabel("Distance (pc)")
#plt.rcParams.update({'font.size': 16})
    