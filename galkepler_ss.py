# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:11:00 2016

@author: john
"""

import cspace_tools
import numpy as np
import matplotlib.pyplot as plt
from math import exp, tan
from astropy.io import fits
from astropy.coordinates import SkyCoord
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleAdiabatic
from galpy.actionAngle import estimateDeltaStaeckel
from galpy.actionAngle import actionAngleStaeckel
from galpy.potential import KeplerPotential, SteadyLogSpiralPotential
from galpy.util import bovy_conversion
from astropy import units
import pickle as cp


#abundances available:
label_tail = "_abund_sme"
data_labels = ["feh_sme",
               "li", "c", "o", "mg",  "al", "si", "k",
               "ca", "sc",  "ti", "v", "cr",  "mn", "co", "ni", "cu", "zn", 
               "y", "ba", "la", "nd", "eu", "rb"]
new_labels =[]
for label in data_labels:
    new_labels.append(label + label_tail)
new_labels[0] = "feh_sme" #because fe_abund_sme is wrong
data_labels = new_labels
#let's forget about the ones we don't care about so much, we could also choose
#to make small deviations in these as flags for non candidates (or rule out any candidates later)
#for x in ["SI_H", "CA_H", "SC_H", "TI_H", "V_H", "CR_H", "MN_H", "NI_H"]:
#    if x in data_labels:
#        data_labels.remove(x)
#retrieve survey data: kepler/galah and mask
hdulist = fits.open('data/sobject_iraf_k2.fits')
cols = hdulist[1].columns
tbdata = hdulist[1].data
mask =  ((tbdata.field("feh_sme") < 0.1) *
        (tbdata.field("feh_sme") > -0.1))
data = np.vstack((tbdata.field(label) for label in data_labels)).T
#mask *= np.all(np.isfinite(data), axis=1)
data = data[mask]

#we will have to grab some metadata ID, GLON/GLAT and whatever else
#is interesting to us after analysing the data
metadata_labels = ["ra", "dec", "rv_sme", "galah_id", "sobject_id"]
metadata = np.vstack((tbdata.field(label) for label in metadata_labels)).T
metadata = metadata[mask]

#check the data
print("data is:", data.shape)
print("metadata is:", metadata.shape)
n = data.shape

#with the FE_H adjustment it is:
m_metric = np.absolute(data.T[0]) #the iron content to start

all_nans = np.sum(np.isnan(data), axis = 1, keepdims = True) # so we can keep track of all the nans
#now get rid of the nans
#print(m_metric[0])
data = np.nan_to_num(data)
for i in range(len(data_labels[1:])): #loop through the other contents measured against iron
#    print(str(m_metric[0]) + " + " + str((data.T[i+1]-data.T[0])[0]) + " gives:")
    m_metric = np.add(m_metric, np.absolute(data.T[i+1]-data.T[0]))
#    print(str(m_metric[0]))
    
#print(str(m_metric[0]) + " - (" + str(data.T[0][0]) + " times " + str(all_nans.T[0][0]) + ")")
m_metric = np.absolute(np.subtract(m_metric, np.absolute(data.T[0]*(all_nans.T[0])))) #one line back we added things that shouldnt be in here

#print(m_metric[0])
m_metric = np.divide(m_metric, np.subtract(n[1], all_nans).T[0])
#print(m_metric[0])
#context

#get things into gal coords if we want (but it's slow)
#def radec_to_gal(ra, dec):
#    coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
#    gal_coord = coord.galactic
#    l = gal_coord.l.degree
#    b = gal_coord.b.degree
#    return l, b
#
#gal_coords = np.vstack(np.array(radec_to_gal(metadata.T[0][i], metadata.T[1][i])) for i in range(data.shape[0]))
#
#plt.plot(gal_coords.T[0], gal_coords.T[1], color = '0.5', marker = ',', ls = 'None')

"""
add condition to reject objects with fewer than 8 or 9 abundance measurements
"""

#three cutoff levels
cutoff_mask = (m_metric < 0.05)
m_metric = m_metric[cutoff_mask]
data = data[cutoff_mask]
metadata = metadata[cutoff_mask]
#gal_coords = gal_coords[cutoff_mask]
#plt.plot(gal_coords.T[0], gal_coords.T[1], 'm.')

#cutoff_mask = (m_metric < 0.5)
#m_metric = m_metric[cutoff_mask]
#data = data[cutoff_mask]
#gal_coords = gal_coords[cutoff_mask]
#plt.plot(gal_coords.T[0], gal_coords.T[1], 'go')
#
#cutoff_mask = (m_metric < 0.03)
#m_metric = m_metric[cutoff_mask]
#data = data[cutoff_mask]
#gal_coords = gal_coords[cutoff_mask]
#plt.plot(gal_coords.T[0], gal_coords.T[1], 'bo')
#plt.show

#make sure we have enough elements for a convincing candidate
elements_mask = (all_nans[cutoff_mask].T[0] < 23)
m_metric = m_metric[elements_mask]
data = data[elements_mask]
metadata = metadata[elements_mask]

#open up distances
print("grabbing gal/kep distance data for ss candidates...")
gk_dist = np.loadtxt('data/dereddened_distances_dr51.csv', dtype = str, delimiter=',')
#clean it up
for i in range(len(gk_dist)):
    for e in range(len(gk_dist[i])):
        gk_dist[i][e] = gk_dist[i][e][2:-1]
#make a dcit with apo_id as key and gk_dist values as a list
dist_dict = []
for i in gk_dist:
    dist_dict.append([i[0], i[1:]])
dist_dict = dict(dist_dict)

#grabbing the rest of the spatial data for these bad boys
spatial_labels = ['ra', 'dec', 'pmra_ucac4', 'e_pmra_ucac4', 'pmdec_ucac4', 'e_pmdec_ucac4', 'barycentric','id_tmass', 'sobject_id']
spatials = fits.open('data/sobject_iraf_general_k2.fits')
tb_spatials = spatials[1].data
spatial_data = np.vstack((tb_spatials.field(label) for label in spatial_labels)).T


#orbit time
print("initialising 3D orbits for solar siblings...")
orbits = []
for sobject in metadata.T[-1]:
    sobject_spatial = spatial_data[(spatial_data.T[-1] == str(int(sobject)))]
    sobject_spatial = sobject_spatial[0]
    rv = metadata[(metadata.T[-1] == sobject)]
    rv = rv[0][2]
    ra = float(sobject_spatial[0])
    dec = float(sobject_spatial[1])
    dmod = float(dist_dict[str(int(sobject))][0])
    dist = (exp((dmod + 5)/5))/1000
    pmRA = float(sobject_spatial[2])
    pmDec = float(sobject_spatial[4])
    Vlos = rv # possibly use a barycentric correction: float(sobject_spatial[6])
    orbits.append(Orbit(vxvv=[ra,dec,dist,pmRA,pmDec,Vlos],radec=True,ro=8.,vo=220., solarmotion = "schoenrich"))

print("initialising solar orbit")
#so we have all wee need to calc an orbit
ts = np.linspace(0,-150,10000)
mwp= MWPotential2014

#sun's orbit for comparison:
sun = Orbit(vxvv=[0, 0, 0, 0, 0, 0],radec=True,ro=8.,vo=220., solarmotion = "schoenrich")
sun.integrate(ts,mwp,method='odeint')
print("integrating all orbits")

#3D orbits
for o in orbits:
    o.integrate(ts,mwp,method='odeint')
    delta_x = o.x(ts) - sun.x(ts)
    delta_y = o.y(ts) - sun.y(ts)
    delta_z = o.z(ts) - sun.z(ts)
    delta = (delta_x**2 + delta_y**2 + delta_z**2)**0.5
    plt.semilogy(sun.time(ts), delta*1000)
    
#2D orbits
#mwp_2D = [i.toPlanar() for i in mwp]
#sun_2D = sun.toPlanar()
#sun_2D.integrate(ts,mwp_2D,method='odeint')
##spiral parameters and defaults
#sp_m = 2    #4    #2
#sp_spv = 50 #20   #
#sp_i = -5*(3.1415/180) #-12    #-5
#sp_x_0 = -120*(3.1415/180)   #-120    #
#sp_a = -sp_m/tan(sp_i)
#sp_gamma = sp_x_0/sp_m
#sp_fr_0 = 0.05  #0.05    #
#ro = 8
#vo = 220
#sp_A = -sp_fr_0 #*((ro*vo)**2) #not sure on this part but adding these makes it too big so leave them
#
#
#sp_component = SteadyLogSpiralPotential(amp = 1, omegas = sp_spv, A = sp_A, alpha = sp_a, gamma = sp_gamma)
#mwp_2D.append(sp_component) # not yet
#for o in orbits:
#    o_2D = o.toPlanar()
#    o_2D.integrate(ts,mwp_2D,method='odeint')
#    delta_x = o_2D.x(ts) - sun_2D.x(ts)
#    delta_y = o_2D.y(ts) - sun_2D.y(ts)
#    delta = (delta_x**2 + delta_y**2)**0.5
#    plt.semilogy(sun_2D.time(ts), delta*1000)
    
    
    

#3D actions
#print("calc mean js")
#ro = 8
#vo = 220
#delta_sun = estimateDeltaStaeckel(MWPotential2014,(sun.R(ts))/ro, (sun.z(ts))/vo)
#aAS_sun = actionAngleStaeckel(pot=MWPotential2014,delta = delta_sun)
#js_sun = aAS_sun((sun.R())/ro,(sun.vR())/vo,(sun.vT())/vo,(sun.z())/ro,(sun.vz())/vo)
#mean_js = np.array([js_sun[0].mean(), js_sun[1].mean(), js_sun[2].mean()])
#
#for o in orbits:
#    try:
#        #quick method: only first instance
#        delta_o = estimateDeltaStaeckel(MWPotential2014,(o.R(ts))/ro,(o.z(ts))/vo)
#    #    plt.plot(js[0] - js_sun[0], ts)
#        aAS = actionAngleStaeckel(pot=MWPotential2014,delta = delta_o)
#        js_o = aAS((o.R())/ro,(o.vR())/vo,(o.vT())/vo,(o.z())/ro,(o.vz())/vo)
#        mean_o_js = np.array([js_o[0].mean(), js_o[1].mean(), js_o[2].mean()])
#        mean_js = np.vstack([mean_js, mean_o_js])
#    except:
#        #problem with this orbit (probs unbound)
#        #let it just be big
#        mean_o_js = np.array([1000*js_sun[0].mean(), 1000*js_sun[1].mean(), 1000*js_sun[2].mean()])
#        mean_js = np.vstack([mean_js, mean_o_js])
#
#print("prepare plot")
#percentage_diffs = np.absolute((mean_js[1:] - mean_js[0]))/mean_js[0]
#
##as per the saved figure
#plt.semilogy(percentage_diffs, 'o')
#plt.title("Percentage Difference of Actions to Solar Orbit")
#plt.axhline(5, linestyle = 'dashed', color = 'k')
#plt.ylabel("Log Percentage")
#plt.xlabel("Object ID")
#plt.xlim(-1,7)
#x_labels = range(0, 7)
#plt.xticks(x_labels, metadata.T[-1].astype(int), rotation = -45)    

#BACK TO CHEM STUFF
#create a mask so that all abundances have info where available
all_abundances_bool = mmm = np.ones((1, 24), dtype = bool)
all_abundances_bool = all_abundances_bool[0]
for i in range(len(data)):
    all_abundances_bool *= (data[i] != 0.0)
abundance_all = []
for i in range(len(data)):
    abundance_all.append(data[i][all_abundances_bool])
abundance_all = np.array(abundance_all)

#find centre of them and calc distance between centre then between any two
ss_centre = np.array([abundance_all.T[i].mean() for i in range(len(abundance_all[0]))])
centre_dist = []
for i in abundance_all:
    x = i - ss_centre
    y = np.absolute(x)
    z = sum(y)/len(ss_centre)
    centre_dist.append(z)
centre_dist = np.array(centre_dist)

pairwise = []
count = 0
for i in abundance_all:
    for j in abundance_all:
        if np.array_equal(i, j) == False:
            count += 1
            x = i - j
            y = np.absolute(x)
            z = sum(y)/len(ss_centre)
            pairwise.append(z)        
#pairwise = np.array(pairwise)
