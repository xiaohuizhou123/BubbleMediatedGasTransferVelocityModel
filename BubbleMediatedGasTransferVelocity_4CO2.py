#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 20:47:49 2023

@author: xzhou
"""

import os
import datetime as dt
import numpy as np
import h5py
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import pi

## define all functions used in for the parameters in bubble mediated gas transfer velocity
def solco2(T):
    r1=8.314
    a=[-60.2409,93.4157,23.3585]
    b=[0.023517,-0.023656,0.0047036]
    a0=[2073.1,125.632,3.6276,0.043219]
    tsx=(T+273.16)/100
    s=a[0]+a[1]/tsx+a[2]*np.log(tsx)+35*(b[0]+b[1]*tsx+b[2]*tsx*tsx)
    sol=np.exp(s)*1000
    al=np.multiply(sol,tsx)/1e5*r1*100
    sc=a0[0]-a0[1]*T+a0[2]*np.power(T,2)-a0[3]*np.power(T,3)
    return sol,al,sc
solco2(np.array([20,30]))

def SW_Psat2(T,S):
    T = T + 273.15
    a = [-5.8002206E+03,1.3914993E+00, -4.8640239E-02, 4.1764768E-05, -1.4452093E-08, 6.5459673E+00]
    Pv_w = np.exp((a[0]/T) + a[1] + a[2]*T + a[3]*T**2 + a[4]*T**3 + a[5]*np.log(T))
    b  = [-4.5818 * 10 ** (-4), -2.0443 * 10**(-6)]
    Pv   = np.multiply(Pv_w,np.exp(b[0]*S+b[1]*S**2))
    return Pv

def SW_Viscosity2(T,S): 
    S = S/1000
    a = [1.5700386464E-01,6.4992620050E+01, -9.1296496657E+01, 4.2844324477E-05,
     1.5409136040E+00,1.9981117208E-02, -9.5203865864E-05, 7.9739318223E+00,
    -7.5614568881E-02, 4.7237011074E-04]
    mu_w = a[3] + 1./(a[0]*(T+a[1])**2+a[2])
    A  = a[4] + a[5]* T + a[6] * T**2
    B  = a[7]+ a[8] * T + a[9]* T**2
    mu = mu_w*(1 + A*S + B*S**2)
    return mu

def SW_Density2(T,S,P):
    P0 = SW_Psat2(T,S)/1E6
    P0[T<100] = 0.101325
    s = S/1000
    a = [9.9992293295E+02, 2.0341179217E-02, -6.1624591598E-03, 2.2614664708E-05, -4.6570659168E-08]
    b = [8.0200240891E+02, -2.0005183488E+00, 1.6771024982E-02, -3.0600536746E-05, -1.6132224742E-05]
    rho_w = a[0] + a[1]*T + a[2]*np.power(T,2) + a[3]*np.power(T,3) + a[4]*np.power(T,4)
    D_rho = b[0]*s + b[1]*s/T + b[2]*s/np.power(T,2) + b[3]*s/np.power(T,3) + b[4]*np.power(s,2)/np.power(T,2)
    rho_sw_sharq=rho_w +D_rho
    c = [5.0792E-04, -3.4168E-06, 5.6931E-08, -3.7263E-10, 1.4465E-12, -1.7058E-15,
    -1.3389E-06, 4.8603E-09, -6.8039E-13]
    d=[-1.1077e-06, 5.5584e-09, -4.2539e-11, 8.3702e-09]
    F_P = np.exp((P-P0)*(c[0] + c[1]*T + c[2]*T**2 + c[3]*T**3 + c[4]*T**4 + c[5]*T**5 + S*(d[0] + d[1]*T + d[2]*T**2)) + 0.5*(P**2-P0**2)*(c[6] + c[7]*T + c[8]*T**3 + d[3]*S))
    rho = rho_sw_sharq*F_P
    return rho

def SW_Kviscosity2(T,S):
    """viscosiity output: kinematic viscosity, m^2/s T = SST, C;S = salinity ppth"""

    P0 = SW_Psat2(T,S)/1E6
    P0[T<100] = 0.101325
    mu  = SW_Viscosity2(T,S)
    rho = SW_Density2(T,S,P0)
    nu  = mu/rho
    return nu

def BubbleGasTransferVelocity(Va,znotm,dc,alpha,sol,Sc,nu):
    """
    Parameters
    ----------
    Va : TYPE
        DESCRIPTION.
    znotm : TYPE
        bubble injection depth.
    dc : TYPE
        the difference between c1 and c2 to do integration.
    T : TYPE
        sea surface temperature.
    S : TYPE
        sea surface salinity.
    alpha: Ostwald solubility
    
    sol: solubility with unit 
    
    Sc: Schimidt Number
    
    Returns
    
    Kbout: unit: m/s
    -------
    None.

    """
    """set up""" 
 #   "%solco2.m and SW_Kviscosity2.m is downloaded from NOAA (ftp://ftp1.esrl.noaa.gov/BLO/Air-Sea/bulkalg/cor3_6/)"
    D=nu/Sc #" modeluclar diffusivity"
 #   " set up bubble range distribution and distribution"
    rm=0.01
    r1=np.arange(1e-5, 1e-3+1e-5,1e-5)
    r2=np.arange(1e-3+1e-5, rm+1e-5,1e-5)
    r=np.arange(1e-5, rm+1e-5,1e-5)
    #r2=np.arange(1e-3+1e-5, rm,1e-5)
    #r=[r1 r2];
    r=np.hstack((r1,r2))
    #only do large bubbles
    beta=3/2
    qr1=np.power(r1,-beta)*(1e-3**(beta-10/3))
    qr2=np.power(r2,-10/3)
    qr=np.hstack((qr1,qr2))
    Vtot_ref_norm=2*pi
    Lr1=r1.shape[0]
    Lr2=r2.shape[0]
    VtotLargeBubbles=np.trapz(4/3*pi*r2**3*r2**(-10/3)/2/pi,x=r2)
    VtotSmallBubbles=np.trapz(4/3*pi*np.power(r1,3)*(r1**(-beta))*(1e-3**(beta-10/3))/2/pi,x=r1)
    coeff=1/(VtotSmallBubbles+VtotLargeBubbles)
    #Vtot_subHinze_norm=np.trapz(np.hstack((r[0:Lr1],4/3*pi*r[0:Lr1]**3.*qr[0:Lr1])))
    #coeff=dc*Vtot_ref_norm/Vtot_subHinze_norm 
    #''' coeffcient to make Va calculated from Lambda (WW3) be equal to gas flux based on physics, based on radius of bubble and dc, here to set dc=0.05''' 
    
    Qr=Va/2/pi*qr
 #   Qr=zeros(size(qr));

#    %% Er
    mu=1.07e-3
    nu=mu/1035
    chi=9.81*(r**3)/(nu**2);#"% equation 7 in DM18"
    yy=10.82/chi
    Ur_clean=2*9.81*r**2./9/nu
    Ur_dirty=2*9.81*r**2/9/nu*((yy**2+2*yy)**(1/2)-yy)
    Ur=Ur_dirty
    Ur[Ur>0.3]=0.3
    Ur_clean[Ur_clean>0.3]=0.3 # Wb(r) in equation 5 and 6

#    "% efficiency coefficient "
    kr=8*np.sqrt(pi*D*Ur/(2*r))#%m/s
    Heqr=4*pi*r*Ur/(3*alpha*kr)
#    % Er=zr./(Heqr+zr);
#    %% Er
 #   % zr=hs; % z_0 ~Hs (Lenain&Melville,2017a)

 #   Er=zeros(size(qr));
#    % efficiency coefficient        
    Er=znotm/(Heqr+znotm)
#    %%
    Vexch=np.trapz((4*pi/3)*(r**3)*Qr*Er,r);
    kbout=1/alpha*Vexch*coeff # unit m/s
    return kbout