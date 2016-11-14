#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 13:08:06 2016

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

fd = open("data/dbscans/gk_ss_cands.pkl", "rb")
gk_candidates = cp.load(fd)
fd.close()
gk_candidates = np.array(gk_candidates)

def generate_rand(measurement, error, measurement_type = None):
    #the type is the important part for converting the measurement
    #the reason for this function is to handle measurement confusion
    if measurement_type == 'mas':
        error = error/3600000
    value = np.random.normal(loc = measurement, scale = error)
    return value

def initialise_orbit(ID):
    cand = gk_candidates[(gk_candidates.T[-1] == int(ID))]
    cand = cand[0]
    ra_mean = cand[0]
    ra_err = 0 #unavailable
    dec_mean = cand[1]
    dec_err = 0
    pmRA_mean = cand[3]
    pmRA_err = cand[6]
    pmDec_mean = cand[4]
    pmDec_err = cand[7]
    
    #calc a random value from gaussian
#    ra = generate_rand(ra_mean, ra_err, measurement_type = 'mas')
#    dec = generate_rand(dec_mean, dec_err, measurement_type = 'mas')
    ra = ra_mean
    dec = dec_mean
    pmRA = generate_rand(pmRA_mean, pmRA_err)
    if pmDec_mean == pmDec_err: #some issue with things being wrong just ignore it
        pmDec = pmDec_mean
    else:
        pmDec = generate_rand(pmDec_mean, pmDec_err)
    #no error for distance in this set
    dist = cand[2]
    
    #VLOS is reliable and stays as is
    Vlos = cand[5]
    orbit = Orbit(vxvv=[ra,dec,dist,pmRA,pmDec,Vlos],radec=True,ro=8.,vo=220., solarmotion = "schoenrich")
    return orbit
    
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
    sp_A = -(1)*sp_fr_0 #*((ro*vo)**2) #not sure on this part but adding these makes it too big so leave them
    sp_component = SteadyLogSpiralPotential(amp = 1, omegas = sp_spv/220, A = sp_A, alpha = sp_a, gamma = sp_gamma)
    mwp_2D.append(sp_component)    
    
ts = np.linspace(0,-150,10000)
mwp = MWPotential2014
sun = Orbit(vxvv=[0, 0, 0, 0, 0, 0],radec=True,ro=8.,vo=220., solarmotion = "schoenrich")
sun.integrate(ts,mwp,method='odeint')
mwp_2D = [i.toPlanar() for i in mwp]   
sun_2D = sun.toPlanar()
sun_2D.integrate(ts,mwp_2D,method='odeint')    

runs = 25
counts = []
for sobject in gk_candidates.T[-1][4:]:
    print("finally calculating distance from sun for 50 iterations " + str(int(sobject)))
    count = 0
#    plots = 0
    arms = 2
    for i in range(runs):
        add_spiral(arms)
        o = initialise_orbit(sobject)
        o_2D = o.toPlanar()
        o_2D.integrate(ts,mwp_2D,method='odeint')
        delta_x = o_2D.x(ts) - sun_2D.x(ts)
        delta_y = o_2D.y(ts) - sun_2D.y(ts)
        delta = (delta_x**2 + delta_y**2)**0.5
        delta = delta*1000
    #    plt.semilogy(sun_2D.time(ts), delta*1000)
        delta_before_4gyr = delta[(sun_2D.time(ts) <= -4)][(sun_2D.time(ts)[(sun_2D.time(ts) <= -4)] >= -5.2)]
        if delta_before_4gyr.min() <= 100:
            count += 1
#        if plots <= 3:
#            plt.semilogy(sun.time(ts), delta)
#        plots += 1
    counts.append(count)
    break
print(runs, arms, counts)    
    
    
    
    
    
    
    
    
    
    
    
    