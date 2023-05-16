#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 22:57:03 2023

@author: Tom
"""


import numpy as np
import os
import sys
module_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(module_dir)
sys.path.append(module_dir)
sys.path.append(main_dir)
from physics_functions import *


def make_sol_dict(outward, inward, units_for_csv=False) :
    m = np.concatenate((outward.t, np.flip(inward.t)))
    r = np.concatenate((outward.y[0], np.flip(inward.y[0])))
    l = np.concatenate((outward.y[1], np.flip(inward.y[1])))
    P = np.concatenate((outward.y[2], np.flip(inward.y[2])))
    T = np.concatenate((outward.y[3], np.flip(inward.y[3])))
    
    rho = density(P, T, mu)
    kappa = opacity(rho, T)
    del_ad = conv_del(P, T, mu)
    del_rad = rad_del(m, l, P, T, mu)
    nabla = np.minimum(del_ad, del_rad)
    eps = epsilon(rho, T)
    eps_pp = epsilon_pp(rho, T)
    eps_CNO = epsilon_CNO(rho, T)
    
    conv_rad = np.empty(len(m), dtype='<U16')
    conv_rad[del_rad >= del_ad] = "convective"
    conv_rad[del_rad < del_ad] = "radiative"
    
    sol_dict = {}
    
    if units_for_csv :
        sol_dict['m (g)'] = m
        sol_dict['r (cm)'] = r
        sol_dict['rho (g cm-3)'] = rho
        sol_dict['T (K)'] = T
        sol_dict['P (dyne cm-2)'] = P
        sol_dict['l (erg s-1)'] = l
        sol_dict['epsilon (erg g-1 s-1)'] = eps
        sol_dict['epsilon_pp (erg g-1 s-1)'] = eps_pp
        sol_dict['epsilon_CNO (erg g-1 s-1)'] = eps_CNO
        sol_dict['kappa (cm2 g-1)'] = kappa
    else :
        sol_dict['m'] = m
        sol_dict['r'] = r
        sol_dict['rho'] = rho
        sol_dict['T'] = T
        sol_dict['P'] = P
        sol_dict['l'] = l
        sol_dict['eps'] = eps
        sol_dict['eps_pp'] = eps_pp
        sol_dict['eps_CNO'] = eps_CNO
        sol_dict['kappa'] = kappa
    sol_dict['del_ad'] = del_ad
    sol_dict['del_rad'] = del_rad
    sol_dict['del'] = nabla
    sol_dict['nature of energy transport'] = conv_rad
    
    return sol_dict