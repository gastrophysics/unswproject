# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:51:47 2016

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle as cp


"""
4 criteria
- all abundances manhattan no adjustment
- important abundances as per ss paper - reject others
- all abundances manhattan with mitschang weighting
- important abundances - reject others
- metric built from PCA
- all and non manhattan
---- maybe add abundances from apokasc?
"""

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
        (tbdata.field("FE_H") < 0.1) *
        (tbdata.field("FE_H") > -0.1) *
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
#plt.plot(metadata.T[4], metadata.T[5], color = '0.5', marker = ',', ls = 'None')


##three cutoff levels
#cutoff_mask = (m_metric < 0.07)
#m_metric = m_metric[cutoff_mask]
#data = data[cutoff_mask]
#metadata = metadata[cutoff_mask]
#plt.plot(metadata.T[4], metadata.T[5], 'm.')
#
##
#cutoff_mask = (m_metric < 0.05)
#m_metric = m_metric[cutoff_mask]
#data = data[cutoff_mask]
#metadata = metadata[cutoff_mask]
#plt.plot(metadata.T[4], metadata.T[5], 'go')

#from here does ss candidates

##last cutoff yields 3 candidates
cutoff_mask = (m_metric < 0.05)
m_metric_a = m_metric[cutoff_mask]
data_a = data[cutoff_mask]
metadata_a = metadata[cutoff_mask]
plt.plot(metadata_a.T[4], metadata_a.T[5], 'bo')
plt.xlim(0, 360)
plt.ylim(-90, 90)
plt.xlabel("Galactic Longitude (deg)")
plt.ylabel("Galactic Latitude (deg)")
plt.title("Apogee Possible Solar Siblings")
#plt.show
#
#apo_candidates_1 = [metadata_a, data_a, m_metric_a]
#fn = open("data/apo_candidates_1.pkl", "wb")
#cp.dump(apo_candidates_1, fn)
#fn.close()
def pairwise_calc(abundance_all):
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
    pairwise = np.array(pairwise)      
    return centre_dist, pairwise
            
            
            
            
            
            
            
            
            
            
            
    
    
