#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:35:21 2016

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

ra, dec = ("17 51 14.02204", "40 04 20.8772") # hours can convert if we want but just do roughly
ra, dec = (267.8083, 40.0725) #degrees
HD162826 = [ra, dec, -16.86, 11.01, 1.7, 33.6/1000]

ra = HD162826[0]
dec = HD162826[1]
pmRa = HD162826[2]
pmDec = HD162826[3]
dist = HD162826[-1]
Vlos = HD162826[-2]

o = Orbit(vxvv=[ra,dec,dist,pmRa,pmDec,Vlos],radec=True,ro=8.,vo=220., solarmotion = "schoenrich")

ts = np.linspace(0,-150,10000)
mwp= MWPotential2014
#schoenrich, hogg, dehnen
sun = Orbit(vxvv=[0, 0, 0, 0, 0, 0],radec=True,ro=8.,vo=220., solarmotion = "schoenrich")
sun.integrate(ts,mwp,method='odeint')
o.integrate(ts,mwp,method='odeint')

ro = 8
vo = 220

#actions
delta_sun = estimateDeltaStaeckel(MWPotential2014,(sun.R(ts))/ro, (sun.z(ts))/vo)
aAS_sun = actionAngleStaeckel(pot=MWPotential2014,delta = delta_sun)
js_sun = aAS_sun((sun.R())/ro,(sun.vR())/vo,(sun.vT())/vo,(sun.z())/ro,(sun.vz())/vo)

delta_o = estimateDeltaStaeckel(MWPotential2014,(o.R(ts))/ro,(o.z(ts))/vo)
aAS = actionAngleStaeckel(pot=MWPotential2014,delta = delta_o)
js_o = aAS((o.R())/ro,(o.vR())/vo,(o.vT())/vo,(o.z())/ro,(o.vz())/vo)

js_sun = np.array(js_sun)
js_o = np.array(js_o)


percentage_diffs = np.absolute((js_sun - js_o))/js_sun

#orbital difference
#delta_x = o.x(ts) - sun.x(ts)
#delta_y = o.y(ts) - sun.y(ts)
#delta_z = o.z(ts) - sun.z(ts)
#delta = (delta_x**2 + delta_y**2 + delta_z**2)**0.5
#plt.semilogy(sun.time(ts), delta*1000)

