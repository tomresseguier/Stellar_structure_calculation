#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:53:39 2023

@author: Tom
"""


import numpy as np
import os
import sys
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy import optimize

module_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(module_dir)
sys.path.append(main_dir)
sys.path.append(module_dir)
from physics_functions import *
from plt_framework import plt_framework

plt_framework()
exec(open(main_dir + "/constants.py").read())
exec(open(main_dir + "/parameters.py").read())


def sol_plotter(outward, inward, fig=None, ax=None, linestyle='-', color='blue', log=False) :
    if fig is None :
        fig, ax = plt.subplots(2, 2)
    ax[0,0].plot(outward.t/Ms, outward.y[0]/Rs, c=color, linestyle=linestyle)
    ax[0,0].plot(inward.t/Ms, inward.y[0]/Rs, c=color, linestyle=linestyle)
    ax[0,0].set_xlabel('m/M$_\odot$')
    ax[0,0].set_ylabel('r/R$_\odot$')
    ax[0,0].set_xlim(0, M/Ms)
    ax[0,1].plot(outward.t/Ms, outward.y[1]/Ls, c=color, linestyle=linestyle)
    ax[0,1].plot(inward.t/Ms, inward.y[1]/Ls, c=color, linestyle=linestyle)
    ax[0,1].set_xlabel('m/M$_\odot$')
    ax[0,1].set_ylabel('l/L$_\odot$')
    ax[0,1].set_xlim(0, M/Ms)
    ax[1,0].plot(outward.t/Ms, outward.y[2], c=color, linestyle=linestyle)
    ax[1,0].plot(inward.t/Ms, inward.y[2], c=color, linestyle=linestyle)
    ax[1,0].set_xlabel('m/M$_\odot$')
    ax[1,0].set_ylabel('P (dyne cm$^{-2}$)')
    ax[1,0].set_xlim(0, M/Ms)
    if log :
        ax[1,0].set_yscale('log')
    ax[1,1].plot(outward.t/Ms, outward.y[3], c=color, linestyle=linestyle)
    ax[1,1].plot(inward.t/Ms, inward.y[3], c=color, linestyle=linestyle)
    ax[1,1].set_xlabel('m/M$_\odot$')
    ax[1,1].set_ylabel('T (K)')
    ax[1,1].set_xlim(0, M/Ms)
    if log :
        ax[1,1].set_yscale('log')
    fig.canvas.draw()
    return fig, ax


def plotter2(outward, inward, vis='default') :
    m = np.concatenate((outward.t, np.flip(inward.t)))
    r = np.concatenate((outward.y[0], np.flip(inward.y[0])))
    l = np.concatenate((outward.y[1], np.flip(inward.y[1])))
    P = np.concatenate((outward.y[2], np.flip(inward.y[2])))
    T = np.concatenate((outward.y[3], np.flip(inward.y[3])))
    
    x_linear = m/Ms
    x_env = -np.log10(1-m/M)
    
    rho = density(P, T, mu)
    kappa = opacity(rho, T)
    del_ad = conv_del(P, T, mu)
    del_rad = rad_del(m, l, P, T, mu)
    nabla = np.minimum(del_ad, del_rad)
    eps = epsilon(rho, T)
    eps_pp = epsilon_pp(rho, T)
    eps_CNO = epsilon_CNO(rho, T)
    
    convection_zone = np.where(del_rad >= del_ad)[0]
    #radiation_zone = np.where(del_rad < del_ad)[0]
    
    fig, ax = plt.subplots(2, 2)
    
    if vis == 'env' :
        ax[0,0].plot(x_env, rho, c='blue')
        ax[0,0].set_xlim(0, 10)
        ax[0,0].set_xlabel(r'-log(1-m/M$_\odot$)')
    else :
        ax[0,0].plot(x_linear, rho, c='blue')
        ax[0,0].set_xlabel(r'm/M$_\odot$')
    ax[0,0].set_ylabel(r'$\rho$ (g cm$^{-3}$)')        
    
    if vis == 'linear' :
        ax[0,1].plot(x_linear, kappa, c='blue')
        ax[0,1].set_xlabel(r'm/M$_\odot$')
    else :
        ax[0,1].plot(x_env, kappa, c='blue')
        ax[0,1].set_xlabel(r'-log(1-m/M$_\odot$)')
        ax[0,1].set_xlim(0, 10)
    ax[0,1].set_ylabel(r'$\kappa$ (cm$^2$/g)')        
    
    if vis == 'linear' :
        ax[1,0].plot(x_linear, del_ad, c='blue', linestyle='dotted', label=r'$\nabla_{ad}$')
        ax[1,0].plot(x_linear, del_rad, c='blue', linestyle='dashed', label=r'$\nabla_{rad}$')
        ax[1,0].plot(x_linear, nabla, c='blue', label=r'$\nabla$')
        ax[1,0].set_xlabel(r'm/M$_\odot$')
    else :
        ax[1,0].plot(x_env, del_ad, c='blue', linestyle='dotted', label=r'$\nabla_{ad}$')
        ax[1,0].plot(x_env, del_rad, c='blue', linestyle='dashed', label=r'$\nabla_{rad}$')
        ax[1,0].plot(x_env, nabla, c='blue', label=r'$\nabla$')
        ax[1,0].set_xlim(0, 10)
        ax[1,0].fill_between(x_env[convection_zone],
                             np.zeros(len(convection_zone)),
                             np.ones(len(convection_zone)),
                             hatch='/', alpha=0.1, facecolor='gray',
                             edgecolor='black', label='convective zone')
        ax[1,0].set_xlabel(r'-log(1-m/M$_\odot$)')
    ax[1,0].set_ylabel(r'$\nabla$')
    ax[1,0].set_ylim(0, 1)
    ax[1,0].legend()
    
    if vis == 'env' :
        ax[1,1].plot(x_env, eps, c='blue')
        ax[1,1].plot(x_env, eps_pp, c='blue', linestyle='dotted', label=r'$\epsilon_{pp}$')
        ax[1,1].plot(x_env, eps_CNO, c='blue', linestyle='dashed', label=r'$\epsilon_{CNO}$')
        ax[1,1].set_xlim(0, 10)
        ax[1,1].set_xlabel(r'-log(1-m/M$_\odot$)')
    else :
        ax[1,1].plot(x_linear, eps, c='blue')
        ax[1,1].plot(x_linear, eps_pp, c='blue', linestyle='dotted', label=r'$\epsilon_{pp}$')
        ax[1,1].plot(x_linear, eps_CNO, c='blue', linestyle='dashed', label=r'$\epsilon_{CNO}$')
        ax[1,1].set_xlabel(r'm/M$_\odot$')
    ax[1,1].set_ylabel(r'$\epsilon$ (erg g$^{-1}$ s$^{-1}$)')     
    ax[1,1].legend()
    
    #ax[1,1].set_yscale('log')
    #ax[1,1].set_xscale('log')
    return fig, ax



def MESA_plotter(MESA_dict, fig=None, ax=None, linestyle='-', color='blue', log=False) :
    m = MESA_dict['mass']
    r = 10**MESA_dict['logR']
    P = 10**MESA_dict['logP']
    T = 10**MESA_dict['logT']
    rho = 10**MESA_dict['logRho']
    eps_pp = MESA_dict['pp']
    eps_CNO = MESA_dict['cno']
    eps_tri_alpha = MESA_dict['tri_alpha']
    eps = eps_pp + eps_CNO + eps_tri_alpha
    X = MESA_dict['x_mass_fraction_H']
    Y = MESA_dict['y_mass_fraction_He']
    Z = MESA_dict['z_mass_fraction_metals']
    mu = 4 / (5*X + 3)
    
    photosphere_L = 9.9954051213112360E-001 * Ls
    l_list = [np.trapz(eps[:i], m[:i]*Ms) for i in range(len(m))]
    l = np.array(l_list) + photosphere_L
    
    if fig is None :
        fig, ax = plt.subplots(2, 2)
    
    ax[0,0].plot(m, r, c=color, linestyle=linestyle)
    ax[0,0].set_xlabel('m/M$_\odot$')
    ax[0,0].set_ylabel('r/R$_\odot$')
    ax[0,0].set_xlim(0, M/Ms)
    
    ax[0,1].plot(m, l/Ls, c=color, linestyle=linestyle)
    ax[0,1].set_xlabel('m/M$_\odot$')
    ax[0,1].set_ylabel('l/L$_\odot$')
    ax[0,1].set_xlim(0, M/Ms)
    
    ax[1,0].plot(m, P, c=color, linestyle=linestyle)
    ax[1,0].set_xlabel('m/M$_\odot$')
    ax[1,0].set_ylabel('P (dyne cm$^{-2}$)')
    ax[1,0].set_xlim(0, M/Ms)
    if log :
        ax[1,0].set_yscale('log')
    ax[1,1].plot(m, T, c=color, linestyle=linestyle)
    ax[1,1].set_xlabel('m/M$_\odot$')
    ax[1,1].set_ylabel('T (K)')
    ax[1,1].set_xlim(0, M/Ms)
    if log :
        ax[1,1].set_yscale('log')
    fig.canvas.draw()
    
    return fig, ax
    


def MESA_plotter2(MESA_dict, fig=None, ax=None, vis='default', color='red') :
    m = MESA_dict['mass']
    r = 10**MESA_dict['logR']
    P = 10**MESA_dict['logP']
    T = 10**MESA_dict['logT']
    rho = 10**MESA_dict['logRho']
    eps_pp = MESA_dict['pp']
    eps_CNO = MESA_dict['cno']
    eps_tri_alpha = MESA_dict['tri_alpha']
    eps = eps_pp + eps_CNO + eps_tri_alpha
    X = MESA_dict['x_mass_fraction_H']
    Y = MESA_dict['y_mass_fraction_He']
    Z = MESA_dict['z_mass_fraction_metals']
    mu = 4 / (5*X + 3)
    
    x_linear = m
    x_env = -np.log10(1-m/M*Ms)
    
    if fig is None :
        fig, ax = plt.subplots(2, 2)
    
    if vis == 'env' :
        ax[0,0].plot(x_env, rho, c=color)
        ax[0,0].set_xlim(0, 10)
        ax[0,0].set_xlabel(r'-log(1-m/M$_\odot$)')
    else :
        ax[0,0].plot(x_linear, rho, c=color)
        ax[0,0].set_xlabel(r'm/M$_\odot$')
    ax[0,0].set_ylabel(r'$\rho$ (g cm$^{-3}$)')
    
    if vis == 'env' :
        ax[1,1].plot(x_env, eps, c=color)
        ax[1,1].plot(x_env, eps_pp, c=color, linestyle='dotted', label=r'$\epsilon_{pp}$')
        ax[1,1].plot(x_env, eps_CNO, c=color, linestyle='dashed', label=r'$\epsilon_{CNO}$')
        ax[1,1].plot(x_env, eps_tri_alpha, c=color, linestyle='dotted', label=r'$\epsilon_{3\alpha}$')
        ax[1,1].set_xlim(0, 10)
        ax[1,1].set_xlabel(r'-log(1-m/M$_\odot$)')
    else :
        ax[1,1].plot(x_linear, eps, c=color)
        ax[1,1].plot(x_linear, eps_pp, c=color, linestyle='dotted', label=r'$\epsilon_{pp}$')
        ax[1,1].plot(x_linear, eps_CNO, c=color, linestyle='dashed', label=r'$\epsilon_{CNO}$')
        ax[1,1].plot(x_linear, eps_tri_alpha, c=color, linestyle='dotted', label=r'$\epsilon_{3\alpha}$')
        ax[1,1].set_xlabel(r'm/M$_\odot$')
    ax[1,1].set_ylabel(r'$\epsilon$ (erg g$^{-1}$ s$^{-1}$)')     
    ax[1,1].legend()
    
    #ax[1,1].set_yscale('log')
    #ax[1,1].set_xscale('log')
    
    fig.canvas.draw()
    return fig, ax





