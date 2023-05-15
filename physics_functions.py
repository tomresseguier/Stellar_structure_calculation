#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:43:36 2023

@author: Tom
"""


import numpy as np
import os
import sys
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt

from utils.utils import opacity_reader
exec(open("./constants.py").read())
exec(open("./parameters.py").read())


table, logR_range, logT_range = opacity_reader('./opacity_tables/GN93hz', table_number=74)
opacity_interpolated = RegularGridInterpolator((logT_range, logR_range), table)


def opacity(rho, T) :
    R = np.maximum( 1e-8, rho / (T*1e-6)**3 )
    T = np.maximum( 10**3.75, T )
    logR = np.log10(R)
    logT = np.log10(T)
    
    if type(logR) == np.ndarray and type(logT) == np.ndarray :
        log_kappa = np.zeros(len(logR))
        for i in range(len(logR)) :
            log_kappa[i] = opacity_interpolated([logT[i], logR[i]])[0]
    else :
        log_kappa = opacity_interpolated([logT, logR])[0]
    kappa = 10**log_kappa
    return kappa


def epsilon_pp(rho, T) :
    T9 = T*1e-9
    T7 = T*1e-7
    psi = 1 # or 1.5 - need to check those
    #f11 = 1
    f11 = np.exp(5.92e-3 * (rho/T7**3)**0.5) # Need to check those
    
    g11 = (1 + 3.82*T9 + 1.51*T9**2 + 0.144*T9**3 - 0.0114*T9**4)
    epsilon_pp = 2.57e4 * psi * f11 * g11 * rho * X**2 * T9**(-2/3) * np.exp(-3.381/T9**(1/3))
    
    return epsilon_pp
    

def epsilon_CNO(rho, T) :
    X_C = 0.173285 * Z
    X_N = 0.053152 * Z
    X_O = 0.482273 * Z
    X_CNO = X_C + X_N + X_O
    
    T9 = T*1e-9
    
    g_14_1 = (1 - 2*T9 + 3.41*T9**2 - 2.43*T9**3)
    epsilon_CNO = 8.24e25 * g_14_1 * X_CNO * X * rho * T9**(-2/3) * np.exp(-15.231*T9**(-1/3) - (T9/0.8)**2)
    
    return epsilon_CNO


def epsilon(rho, T) :
    epsilon = epsilon_pp(rho, T) + epsilon_CNO(rho, T)
    return epsilon


def Pressure(rho, T, mu) :
    P = rho*Na*k*T/mu + a*T**4/3
    return P


def density(P, T, mu) :
    #rho = mu/(Na*k) * (P/T - a*T**3/3)
    rho = np.maximum( 0, mu/(Na*k) * (P/T - a*T**3/3) )
    return rho


def conv_del(P, T, mu) :
    rho = density(P, T, mu)    
    beta = rho*Na*k*T/mu / P
    del_ad = 2*(4-3*beta) / (32-24*beta-3*beta**2)
    return del_ad


def rad_del(m, l, P, T, mu) :
    rho = density(P, T, mu)    
    del_rad = 3*P*opacity(rho, T)*l / (16*np.pi*a*c*T**4*G*m)
    return del_rad







