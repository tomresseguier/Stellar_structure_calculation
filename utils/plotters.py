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
import matplotlib.ticker as ticker

module_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(module_dir)
sys.path.append(main_dir)
sys.path.append(module_dir)
from physics_functions import *
from plt_framework import plt_framework

plt_framework()
exec(open(main_dir + "/constants.py").read())
exec(open(main_dir + "/parameters.py").read())


def sol_plotter(outward, inward, fig=None, ax=None, linestyle='-', color='blue', log=False, label='converged solution') :
    if fig is None :
        fig, ax = plt.subplots(2, 2)
        
    ax[0,0].plot(outward.t/Ms, outward.y[0]/Rs, c=color, linestyle=linestyle)
    ax[0,0].plot(inward.t/Ms, inward.y[0]/Rs, c=color, linestyle=linestyle, label=label)
    ax[0,0].set_xlabel('m/M$_\odot$')
    ax[0,0].set_ylabel('r/R$_\odot$')
    ax[0,0].set_xlim(0, M/Ms)
    ax[0,0].legend()
    
    ax[0,1].plot(outward.t/Ms, outward.y[1]/Ls, c=color, linestyle=linestyle)
    ax[0,1].plot(inward.t/Ms, inward.y[1]/Ls, c=color, linestyle=linestyle, label=label)
    ax[0,1].set_xlabel('m/M$_\odot$')
    ax[0,1].set_ylabel('l/L$_\odot$')
    ax[0,1].set_xlim(0, M/Ms)
    ax[0,1].legend()
    
    ax[1,0].plot(outward.t/Ms, outward.y[2], c=color, linestyle=linestyle)
    ax[1,0].plot(inward.t/Ms, inward.y[2], c=color, linestyle=linestyle, label=label)
    ax[1,0].set_xlabel('m/M$_\odot$')
    ax[1,0].set_ylabel('P (dyne cm$^{-2}$)')
    ax[1,0].set_xlim(0, M/Ms)
    if log :
        ax[1,0].set_yscale('log')
    ax[1,0].legend()
    
    ax[1,1].plot(outward.t/Ms, outward.y[3], c=color, linestyle=linestyle)
    ax[1,1].plot(inward.t/Ms, inward.y[3], c=color, linestyle=linestyle, label=label)
    ax[1,1].set_xlabel('m/M$_\odot$')
    ax[1,1].set_ylabel('T (K)')
    ax[1,1].set_xlim(0, M/Ms)
    if log :
        ax[1,1].set_yscale('log')
    ax[1,1].legend()
    
    fig.canvas.draw()
    return fig, ax


def plotter2(sol_dict, vis='default') :
    m = sol_dict['m']
    r = sol_dict['r']
    l = sol_dict['l']
    P = sol_dict['P']
    T = sol_dict['T']
    
    x_linear = m/Ms
    x_env = -np.log10(1-m/M)
    
    rho = sol_dict['rho']
    kappa = sol_dict['kappa']
    del_ad = sol_dict['del_ad']
    del_rad = sol_dict['del_rad']
    nabla = sol_dict['del']
    eps = sol_dict['eps']
    eps_pp = sol_dict['eps_pp']
    eps_CNO = sol_dict['eps_CNO']
    
    convection_zone = np.where(del_rad >= del_ad)[0]
    #radiation_zone = np.where(del_rad < del_ad)[0]
    
    fig, ax = plt.subplots(2, 2)
    
    if vis == 'env' :
        ax[0,0].plot(x_env, rho, c='blue')
        ax[0,0].set_xlim(0, 10)
        ax[0,0].set_xlabel(r'-log(1-m/M$_*$)')
    else :
        ax[0,0].plot(x_linear, rho, c='blue')
        ax[0,0].set_xlabel(r'm/M$_\odot$')
        ax[0,0].set_xlim(0, M/Ms)
    ax[0,0].set_ylabel(r'$\rho$ (g cm$^{-3}$)')        
    
    if vis == 'linear' :
        ax[0,1].plot(x_linear, kappa, c='blue')
        ax[0,1].set_xlabel(r'm/M$_\odot$')
        ax[0,1].set_xlim(0, M/Ms)
    else :
        ax[0,1].plot(x_env, kappa, c='blue')
        ax[0,1].set_xlabel(r'-log(1-m/M$_*$)')
        ax[0,1].set_xlim(0, 10)
    
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax[0,1].yaxis.set_major_formatter(formatter)
    
    ax[0,1].set_ylabel(r'$\kappa$ (cm$^2$/g)')      
    ax[0,1].set_yscale('log')
    
    if vis == 'linear' :
        ax[1,0].plot(x_linear, nabla, c='blue', label=r'$\nabla$')
        ax[1,0].plot(x_linear, del_ad, c='blue', linestyle='dotted', label=r'$\nabla_{ad}$')
        ax[1,0].plot(x_linear, del_rad, c='blue', linestyle='dashed', label=r'$\nabla_{rad}$')
        ax[1,0].set_xlim(0, M/Ms)
        ax[1,0].set_xlabel(r'm/M$_\odot$')
    else :
        ax[1,0].plot(x_env, nabla, c='blue', label=r'$\nabla$')
        ax[1,0].plot(x_env, del_ad, c='blue', linestyle='dotted', label=r'$\nabla_{ad}$')
        ax[1,0].plot(x_env, del_rad, c='blue', linestyle='dashed', label=r'$\nabla_{rad}$')
        ax[1,0].set_xlim(0, 10)
        ax[1,0].fill_between(x_env[convection_zone],
                             np.zeros(len(convection_zone)),
                             np.ones(len(convection_zone)),
                             hatch='/', alpha=0.1, facecolor='gray',
                             edgecolor='black', label='convective zone')
        ax[1,0].set_xlabel(r'-log(1-m/M$_*$)')
    ax[1,0].set_ylabel(r'$\nabla$')
    ax[1,0].set_ylim(0, 1)
    ax[1,0].legend()
    
    if vis == 'env' :
        ax[1,1].plot(x_env, eps, c='blue', label=r'$\epsilon$')
        ax[1,1].plot(x_env, eps_pp, c='blue', linestyle='dotted', label=r'$\epsilon_{pp}$')
        ax[1,1].plot(x_env, eps_CNO, c='blue', linestyle='dashed', label=r'$\epsilon_{CNO}$')
        ax[1,1].set_xlim(0, 10)
        ax[1,1].set_xlabel(r'-log(1-m/M$_*$)')
    else :
        ax[1,1].plot(x_linear, eps, c='blue', label=r'$\epsilon$')
        ax[1,1].plot(x_linear, eps_pp, c='blue', linestyle='dotted', label=r'$\epsilon_{pp}$')
        ax[1,1].plot(x_linear, eps_CNO, c='blue', linestyle='dashed', label=r'$\epsilon_{CNO}$')
        ax[1,1].set_xlabel(r'm/M$_\odot$')
        ax[1,1].set_xlim(0, M/Ms)
    ax[1,1].set_ylabel(r'$\epsilon$ (erg g$^{-1}$ s$^{-1}$)')     
    ax[1,1].legend()
    
    #ax[1,1].set_yscale('log')
    #ax[1,1].set_xscale('log')
    return fig, ax



def MESA_plotter(MESA_profile, fig=None, ax=None, linestyle='-', color='blue', log=False) :
    m = MESA_profile.mass
    r = 10**MESA_profile.logR
    P = 10**MESA_profile.logP
    T = 10**MESA_profile.logT
    rho = 10**MESA_profile.logRho
    eps_pp = MESA_profile.pp
    eps_CNO = MESA_profile.cno
    eps_tri_alpha = MESA_profile.tri_alpha
    eps = eps_pp + eps_CNO + eps_tri_alpha
    X = MESA_profile.x_mass_fraction_H
    Y = MESA_profile.y_mass_fraction_He
    Z = MESA_profile.z_mass_fraction_metals
    mu = 4 / (5*X + 3)
    
    #photosphere_L = 9.9954051213112360E-001 * Ls
    #l_list = [np.trapz(eps[:i], m[:i]*Ms) for i in range(len(m))]
    #l = np.array(l_list) + photosphere_L
    
    if fig is None :
        fig, ax = plt.subplots(2, 2)
    
    ax[0,0].plot(m, r, c=color, linestyle=linestyle)
    ax[0,0].set_xlabel('m/M$_\odot$')
    ax[0,0].set_ylabel('r/R$_\odot$')
    ax[0,0].set_xlim(0, M/Ms)
    
    #ax[0,1].plot(m, l/Ls, c=color, linestyle=linestyle)
    #ax[0,1].set_xlabel('m/M$_\odot$')
    #ax[0,1].set_ylabel('l/L$_\odot$')
    #ax[0,1].set_xlim(0, M/Ms)
    
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
    


def MESA_plotter2(MESA_profile, fig=None, ax=None, vis='default', color='red') :
    m = MESA_profile.mass
    r = 10**MESA_profile.logR
    P = 10**MESA_profile.logP
    T = 10**MESA_profile.logT
    rho = 10**MESA_profile.logRho
    eps_pp = MESA_profile.pp
    eps_CNO = MESA_profile.cno
    eps_tri_alpha = MESA_profile.tri_alpha
    eps = eps_pp + eps_CNO + eps_tri_alpha
    X = MESA_profile.x_mass_fraction_H
    Y = MESA_profile.y_mass_fraction_He
    Z = MESA_profile.z_mass_fraction_metals
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
        ax[0,0].set_xlim(0, M/Ms)
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
        ax[1,1].set_xlim(0, M/Ms)
    ax[1,1].set_ylabel(r'$\epsilon$ (erg g$^{-1}$ s$^{-1}$)')     
    ax[1,1].legend()
    
    #ax[1,1].set_yscale('log')
    #ax[1,1].set_xscale('log')
    
    fig.canvas.draw()
    return fig, ax


def MESA_plotter3(sol_dict, MESA_profile) :
    m = sol_dict['m']
    r = sol_dict['r']
    P = sol_dict['P']
    T = sol_dict['T']
    
    m_MESA = MESA_profile.mass
    r_MESA = 10**MESA_profile.logR
    P_MESA = 10**MESA_profile.logP
    T_MESA = 10**MESA_profile.logT
    
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
    fig.tight_layout(pad=1.0)
    
    ax[0].plot(m/Ms, r/Rs, c='blue', label='this work')
    ax[0].plot(m_MESA, r_MESA, c='red', linestyle='dashed', label='MESA')
    ax[0].set_xlabel('m/M$_\odot$')
    ax[0].set_ylabel('r/R$_\odot$')
    ax[0].set_xlim(0, M/Ms)
    ax[0].legend()
    
    ax[1].plot(m/Ms, P, c='blue', label='this work')
    ax[1].plot(m_MESA, P_MESA, c='red', linestyle='dashed', label='MESA')
    ax[1].set_xlabel('m/M$_\odot$')
    ax[1].set_ylabel('P (dyne cm$^{-2}$)')
    ax[1].set_xlim(0, M/Ms)
    ax[1].legend()
    
    ax[2].plot(m/Ms, T, c='blue', label='this work')
    ax[2].plot(m_MESA, T_MESA, c='red', linestyle='dashed', label='MESA')
    ax[2].set_xlabel('m/M$_\odot$')
    ax[2].set_ylabel('T (K)')
    ax[2].set_xlim(0, M/Ms)
    ax[2].legend()
    return fig, ax


def MESA_eps_plotter(sol_dict, MESA_profile, log=False) :
    m = sol_dict['m']
    eps = sol_dict['eps']
    eps_pp = sol_dict['eps_pp']
    eps_CNO = sol_dict['eps_CNO']
    
    m_MESA = MESA_profile.mass

    eps_pp_MESA = MESA_profile.pp
    eps_CNO_MESA = MESA_profile.cno
    eps_tri_alpha_MESA = MESA_profile.tri_alpha
    eps_MESA = eps_pp_MESA + eps_CNO_MESA + eps_tri_alpha_MESA
    
    fig, ax = plt.subplots()
    ax.plot(m/Ms, eps, c='blue', label=r"this work's $\epsilon$")
    ax.plot(m/Ms, eps_pp, c='blue', linestyle='dotted', label=r"this work's $\epsilon_{pp}$")
    ax.plot(m/Ms, eps_CNO, c='blue', linestyle='dashed', label=r"this work's $\epsilon_{CNO}$")
    
    ax.plot(m_MESA, eps_MESA, c='red', label=r"MESA's $\epsilon$")
    ax.plot(m_MESA, eps_pp_MESA, c='red', linestyle='dotted', label=r"MESA's $\epsilon_{pp}$")
    ax.plot(m_MESA, eps_CNO_MESA, c='red', linestyle='dashed', label=r"MESA's $\epsilon_{CNO}$")
    #ax.plot(m_MESA, eps_tri_alpha_MESA, c='red', linestyle='dashdot', label=r"MESA's $\epsilon_{3\alpha}$")
    
    ax.set_xlabel('m/M$_\odot$')
    ax.set_ylabel(r'$\epsilon$ (erg g$^{-1}$ s$^{-1}$)')     
    ax.set_xlim(0, M/Ms)
    ax.legend()
    if log :
        ax.set_yscale('log')
    return fig, ax


def MESA_mf_plotter(MESA_profile) :
    m_MESA = MESA_profile.mass
    X_MESA = MESA_profile.x_mass_fraction_H
    Y_MESA = MESA_profile.y_mass_fraction_He
    Z_MESA = MESA_profile.z_mass_fraction_metals
    
    fig, ax = plt.subplots()
    ax.plot(m_MESA, X_MESA, c='red', label=r"X")
    ax.plot(m_MESA, Y_MESA, c='red', linestyle='dashed', label=r"Y")
    ax.plot(m_MESA, Z_MESA, c='red', linestyle='dotted', label=r"Z")
    
    ax.set_xlabel('m/M$_\odot$')
    ax.set_ylabel(r'mass fractions')     
    ax.set_xlim(0, M/Ms)
    ax.legend(loc='upper right')
    return fig, ax



def eps_plotter() :
    fig, ax = plt.subplots()
    T_log = np.logspace(6 + np.log10(4), 8, 1000)
    rho = 1 #density(Pc_guess, T_log, mu)
    eps_pp = epsilon_pp(rho, T_log)
    eps_CNO = epsilon_CNO(rho, T_log)
    ax.plot(T_log, eps_pp, c='blue', linestyle=(0, (1, 1)), label=r'$\epsilon_{pp}$')
    ax.plot(T_log, eps_CNO, c='blue', linestyle='dashed', label=r'$\epsilon_{CNO}$')
    #ax.plot(T_log, eps_pp + eps_CNO, c='blue', linestyle='-')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('T (K)')
    ax.set_ylabel(r'$\epsilon$ (erg g$^{-1}$ s$^{-1}$)')
    ax.set_xlim(10**(6 + np.log10(4)), 10**8)
    ax.legend()
    fig.savefig("eps.png")
    return fig, ax


def opacity_plotter(cmap='inferno') :
    #x = np.linspace(10**3.75, 10**8.7, 100)
    #y = np.linspace((10**3.75/10**6)**3*1e-8, (10**8.7/10**6)**3*10, 100)
    logT = np.linspace(3.75, 8.7, 400)
    logR = np.linspace(-8, 1, 200)
    
    LogR, LogT = np.meshgrid(logR, logT)
    
    Kappa = np.zeros(LogT.shape)
    for i in range(LogT.shape[0]) :
        for j in range(LogT.shape[1]) :
            #Z[i,j] = opacity(X[i,j], Y[i,j])
            Kappa[i,j] = opacity_interpolated([LogT[i,j], LogR[i,j]])
    
    fig, ax = plt.subplots()
    im = ax.imshow(Kappa, extent=[LogR.min(), LogR.max(), LogT.min(), LogT.max()], origin='lower', cmap=cmap)
    ax.set_xlabel(r'log$_{10}$(R)')
    ax.set_ylabel(r'log$_{10}$(T)')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.02625, pad=0.02)
    fig.colorbar(im, cax=cbar.ax, format='%.2f')
    cbar.set_label(r'log$_{10}$($\kappa$)')
    
    fig.savefig("opacity.png")
    return fig, ax
















