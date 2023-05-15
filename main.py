#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:02:08 2023

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

from physics_functions import *
from utils.plotters import *
from utils.MESA_profile_reader import MESA_profile_reader
from utils.plt_framework import plt_framework
plt_framework()

exec(open("./constants.py").read())
exec(open("./parameters.py").read())



step = M / N_steps

# Values for the Sun
R_guess = Rs
L_guess = Ls
Pc_guess = 2.65e17
Tc_guess = 1.5e7
Sun_guess = [R_guess, L_guess, Pc_guess, Tc_guess]

########## Constant density model guess ##########
Pc_guess = 3/(8*np.pi) * (G*M**2)/R_guess**4
Tc_guess = G*M*mu/(2*R_guess*Na*k)
cste_density_guess = [R_guess, L_guess, Pc_guess, Tc_guess]


def load1(mr, Pc, Tc) :
    rho_c = density(Pc, Tc, mu)
    r = (3*mr/(4*np.pi*rho_c))**(1/3)
    lr = epsilon(rho_c, Tc) * mr
    Pr = Pc - 3*G/(8*np.pi) * (4*np.pi/3*rho_c)**(4/3) * mr**(2/3)
    Tr = ( Tc**4 - 1/(2*a*c) * (3/(4*np.pi))**(2/3) * opacity(rho_c, Tc)*epsilon(rho_c, Tc)*rho_c**(4/3)*mr**(2/3) )**(1/4) # radiative core
    return [r, lr, Pr, Tr]

def load2(R, L) : #T_eff
    T_eff = ( L / (4*np.pi*R**2*sb) )**(1/4)
    #L = 4*np.pi*R**2*sb*T_eff**4
    #P_R = G*M/R**2 * 2/3 * 1/kappa_mean
    def T(tau) :
        T = T_eff * (3/4*(tau + 2/3))**(1/4)
        return T
    def dP(tau, P) :
        dP = G*M/R**2 * 1/opacity(density(P[0], T(tau), mu), T(tau))
        return dP
    def P(tau) :
        sol = solve_ivp(dP, [0, tau], [0])
        return sol.y[0][-1]
    P_R = P(2/3)
    T_R = T(2/3)
    return [R, L, P_R, T_R]


def derivatives(m, y) :
    r, l, P, T = y
    rho = density(P, T, mu)    
    del_rad = rad_del(m, l, P, T, mu)
    del_ad = conv_del(P, T, mu)
    #del_ad = 0.4
    if del_rad <= del_ad :
        nabla = del_rad
    else :
        nabla = del_ad
    
    dr = 1/(4*np.pi*r**2*rho)
    dl = epsilon(rho, T)
    dP = -G*m/(4*np.pi*r**4)
    dT = -G*m*T/(4*np.pi*r**4*P) * nabla
    return [dr, dl, dP, dT]


def shoot(x) :
    R, L, Pc, Tc = x
    outward_eval = np.linspace(mr, shooting_fraction*M, int(shooting_fraction*N_steps))
    envelope_eval = M - np.logspace(np.log10(M*1e-10), np.log10(0.1*M), int(0.1*N_steps))
    switch = np.where(envelope_eval[:-1] - envelope_eval[1:] > step)[0][0]
    rest_start = envelope_eval[switch]
    envelope_eval = envelope_eval[:switch]
    N_steps_rest = round( (rest_start - shooting_fraction*M) / step )
    rest_eval = np.linspace(rest_start, shooting_fraction*M, N_steps_rest)
    inward_eval = np.concatenate( (envelope_eval, rest_eval) )    
    #print('switch = ', switch)
    #print('number of steps in envelope eval: ', int(0.1*N_steps))
    #print('extent: ', -np.log(1-envelope_eval[switch-1]/M))
    
    outward = solve_ivp(derivatives, [mr, shooting_fraction*M], load1(mr, Pc, Tc), t_eval=outward_eval)
    inward = solve_ivp(derivatives, [M, shooting_fraction*M], load2(R, L), t_eval=inward_eval)
    return outward, inward

def loss_multi(outward, inward) :
    score_multi = [ 2*(outward.y[i,-1] - inward.y[i,-1])/(outward.y[i,-1] + inward.y[i,-1]) for i in range(4) ]
    return score_multi

def shoot_and_score_root(x) :
    outward, inward = shoot(x)
    score = loss_multi(outward, inward)
    return score

sol = optimize.root(shoot_and_score_root, Sun_guess)
outward, inward = shoot(sol.x)
outward_guess, inward_guess = shoot(Sun_guess)

fig0, ax0 = sol_plotter(outward_guess, inward_guess, linestyle='dashed', log=False)

fig1, ax1 = sol_plotter(outward, inward, linestyle='-', log=False)
#fig1, ax1 = sol_plotter(outward_guess, inward_guess, fig=fig1, ax=ax1, linestyle='dotted', color='black', log=False)
fig2, ax2 = plotter2(outward, inward, vis='default')



########## Comparing with MESA ZAMS results ##########
profile_dir = "./MESA_profiles/"
profile_name = "1.1M_at_ZAMS.data"
profile_path = profile_dir + profile_name
MESA_dict = MESA_profile_reader(profile_path)
fig3, ax3 = MESA_plotter(MESA_dict, fig=fig1, ax=ax1, linestyle='-', color='red', log=False)
#fig3, ax3 = MESA_plotter(MESA_dict)
fig4, ax4 = MESA_plotter2(MESA_dict, fig=fig2, ax=ax2, color='red')




########## Plotting the energy generation rates ##########
fig, ax = plt.subplots()
T_log = np.logspace(6, 8, 1000)
rho = 1 #density(Pc_guess, T_log, mu)
eps_pp = epsilon_pp(rho, T_log)
eps_CNO = epsilon_CNO(rho, T_log)
ax.plot(T_log, eps_pp)
ax.plot(T_log, eps_CNO)
ax.set_xscale('log')
ax.set_yscale('log')












#def loss_partials(outward, inward) :    
#    score = np.sum( ( 2*(outward.y[:,-1] - inward.y[:,-1])/(outward.y[:,-1] + inward.y[:,-1]) )**2 )**0.5
#    outward_derivs = np.array( derivatives(shooting_fraction*M, outward.y[:,-1]) )
#    inward_derivs = np.array( derivatives(shooting_fraction*M, inward.y[:,-1]) )
#    score_derivs = np.sum( ( 2*(outward_derivs - inward_derivs)/(outward_derivs + inward_derivs) )**2 )**0.5
#    return (score**2 + score_derivs**2)**0.5

#def loss(outward, inward) :
#    score = np.sum( ( 2*(outward.y[:,-1] - inward.y[:,-1])/(outward.y[:,-1] + inward.y[:,-1]) )**2 )**0.5
#    return score

#def shoot_and_score(x) :
#    outward, inward = shoot(x)
#    score = loss(outward, inward)
#    return score

#sol = optimize.minimize(shoot_and_score, Sun_guess)






