# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:51:47 2016

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

#abundances available:
data_labels = ["FE_H",
               "AL_H", "CA_H", "C_H", "K_H",  "MG_H", "MN_H", "NA_H",
               "NI_H", "N_H",  "O_H", "SI_H", "S_H",  "TI_H", "V_H"]
#let's forget about the ones we don't care about so much, we could also choose
#to make small deviations in these as flags for non candidates (or rule out any candidates later)
#for x in ["SI_H", "CA_H", "SC_H", "TI_H", "V_H", "CR_H", "MN_H", "NI_H"]:
#    if x in data_labels:
#        data_labels.remove(x)
#retrieve survey data: APOGEE and mask
hdulist = fits.open('data/results-unregularized-matched.fits')
cols = hdulist[1].columns
tbdata = hdulist[1].data
mask = ((tbdata.field("TEFF_ASPCAP") > 3500.) *
        (tbdata.field("TEFF_ASPCAP") < 5500.) *
        (tbdata.field("FE_H") < 0.2) *
        (tbdata.field("FE_H") > -0.2) *
        (tbdata.field("LOGG_ASPCAP") > 0.) *
        (tbdata.field("LOGG_ASPCAP") < 3.9)) # MKN recommendation
metadata_check_labels = ["RA", "DEC", "TEFF_ASPCAP", "LOGG_ASPCAP"]
metadata_check = np.vstack((tbdata.field(label) for label in metadata_check_labels)).T
mask *= np.all(np.isfinite(metadata_check), axis=1)
data = np.vstack((tbdata.field(label) for label in data_labels)).T
mask *= np.all(np.isfinite(data), axis=1)
data = data[mask]

#we will have to grab some metadata ID, GLON/GLAT and whatever else
#is interesting to us after analysing the data
metadata_labels = ["RA", "DEC", "PMRA", "PMDEC", "GLON", "GLAT", "VHELIO_AVG", "TEFF_ASPCAP", "LOGG_ASPCAP", "FIELD", "APOGEE_ID"]
metadata = np.vstack((tbdata.field(label) for label in metadata_labels)).T
metadata = metadata[mask]

#check the data
print("data is:", data.shape)
print("metadata is:", metadata.shape)
n = data.shape

#metric_column = vactor sum the "data" columns
#n on denominator not really needed on one dataset, only there if we are to compare to other things
#without the FE_H adjustment it is simple:
mmetric = np.sum(np.absolute(data), 1)/n[1]
print(mmetric.shape)

#with the FE_H adjustment it is:
m_metric = np.absolute(data.T[0]) #the iron content to start
for i in range(len(data_labels[1:])): #loop through the other contents measured against iron
    m_metric = np.add(m_metric, np.absolute(data.T[i+1]-data.T[0])) 
m_metric = m_metric/n[1]


#plot different cutoffs
#context
plt.plot(metadata.T[4], metadata.T[5], color = '0.5', marker = ',', ls = 'None')


#three cutoff levels
#cutoff_mask = (m_metric < 0.07)
#m_metric = m_metric[cutoff_mask]
#data = data[cutoff_mask]
#metadata = metadata[cutoff_mask]
#plt.plot(metadata.T[2], metadata.T[3], 'm.')

#cutoff_mask = (m_metric < 0.05)
#m_metric = m_metric[cutoff_mask]
#data = data[cutoff_mask]
#metadata = metadata[cutoff_mask]
#plt.plot(metadata.T[2], metadata.T[3], 'go')

#last cutoff yields 3 candidates
cutoff_mask = (m_metric < 0.045)
m_metric = m_metric[cutoff_mask]
data = data[cutoff_mask]
metadata = metadata[cutoff_mask]
plt.plot(metadata.T[4], metadata.T[5], 'bo')
plt.show
"""
at this point we find the candidate distances and look at their info
for whatever reason one of the objects has two different readings in the
distance file... ask sarah?
"""
#let us load up some more info (distances from sarah)

#hdudist = fits.open('data/allStar+-v603.150209.fits')
#distances = hdudist[1].data.field('DISO').T[1].T
#ap_id = hdudist[1].data.field('APOGEE_ID')
#dist_table = np.vstack([ap_id, distances]).T
#
##
#candidate_ids = metadata.T[-1]
#for ID in candidate_ids:
#    for i in dist_table:
#        if ID == i[0]:
#            print(i)
#
#for ID in candidate_ids:
#    for i in metadata:
#        if ID == i[-1]:
#            print(i)
#        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
