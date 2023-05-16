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
import pandas as pd
import mesa_reader

from physics_functions import *
from utils.plotters import *
from utils.utils import *
from utils.MESA_profile_reader import MESA_profile_reader
from utils.plt_framework import plt_framework
plt_framework()

# load the constants and parameters of the model
exec(open("./constants.py").read())
exec(open("./parameters.py").read())


# Number of steps where the solution will be evaluated in the integration's output, mostly useful for plots
step = M / N_steps

# Initial guess
first_guess = [R_guess, L_guess, Pc_guess, Tc_guess]

# function that calculates and returns the center boundary conditions
def load1(mr, Pc, Tc) :
    rho_c = density(Pc, Tc, mu)
    r = (3*mr/(4*np.pi*rho_c))**(1/3)
    lr = epsilon(rho_c, Tc) * mr
    Pr = Pc - 3*G/(8*np.pi) * (4*np.pi/3*rho_c)**(4/3) * mr**(2/3)
    Tr = ( Tc**4 - 1/(2*a*c) * (3/(4*np.pi))**(2/3) * opacity(rho_c, Tc)*epsilon(rho_c, Tc)*rho_c**(4/3)*mr**(2/3) )**(1/4) # radiative core
    return [r, lr, Pr, Tr]

# function that calculates and returns the edge's boundary conditions
def load2(R, L) :
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
        sol = solve_ivp(dP, [0, tau], [2.])
        return sol.y[0][-1]
    P_R = P(2/3)
    T_R = T(2/3)
    return [R, L, P_R, T_R]


# function that calculates and returns the derivatives for a given m
# used by the integrator
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


# function that integrates the system outwards and inwards until shooting_fraction after loading the boundary conditions
def shoot(x) :
    R, L, Pc, Tc = x
    # All this bit is not crucial, it just allows for more points to be evaluated near the surface
    # Useful for the plots that use the -log(1-m/M) scale
    outward_eval = np.linspace(mr, shooting_fraction*M, int(shooting_fraction*N_steps))
    envelope_eval = M - np.logspace(np.log10(M*1e-10), np.log10(0.1*M), int(0.1*N_steps))
    switch = np.where(envelope_eval[:-1] - envelope_eval[1:] > step)[0][0]
    rest_start = envelope_eval[switch]
    envelope_eval = envelope_eval[:switch]
    N_steps_rest = round( (rest_start - shooting_fraction*M) / step )
    rest_eval = np.linspace(rest_start, shooting_fraction*M, N_steps_rest)
    inward_eval = np.concatenate( (envelope_eval, rest_eval) )    
    
    # Heart of the shoot function: the integration
    outward = solve_ivp(derivatives, [mr, shooting_fraction*M], load1(mr, Pc, Tc), t_eval=outward_eval)
    inward = solve_ivp(derivatives, [M, shooting_fraction*M], load2(R, L), t_eval=inward_eval)
    return outward, inward

# function that calculates the 'score' of an outward and an inward solution
def loss_multi(outward, inward) :
    score_multi = [ 2*(outward.y[i,-1] - inward.y[i,-1])/(outward.y[i,-1] + inward.y[i,-1]) for i in range(4) ]
    return score_multi

# function that calculates the 'score' for a given guess and that we will want to find the zero of
def shoot_and_score_root(x) :
    outward, inward = shoot(x)
    score = loss_multi(outward, inward)
    return score


# Now we find the zero of shoot_and_score_root, starting with first_guess
sol = optimize.root(shoot_and_score_root, first_guess)

# Print some results
print('##############################')
print('Final parameters for the star:')
print('R_* = ', sol.x[0])
print('    = ', sol.x[0]/Rs, 'R_sun')
print('L_* = ', sol.x[1])
print('    = ', sol.x[1]/Ls, 'L_sun')
print('P_c = ', sol.x[2])
print('T_c = ', sol.x[3])
print('##############################')

print('########################################')
print('Additional parameters for the star:')
print('g = ', G*M/sol.x[0]**2)
print('T_eff = ', load2(sol.x[0], sol.x[1])[-1])
print('########################################')

# Calculate the profiles of the dependent variables from the final solution we found
outward, inward = shoot(sol.x)
# Calculate the profiles of the dependent variables from the initial guess to be compared with the final solution
outward_guess, inward_guess = shoot(first_guess)
# Create convenient dictionaries with the results
sol_dict = make_sol_dict(outward, inward)
guess_dict = make_sol_dict(outward_guess, inward_guess)


# Create the figure directory
fig_dir = "./figures/"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    print("figures directory created")

# This function plots the profiles of r, l, P, T
fig1, ax1 = sol_plotter(outward, inward, linestyle='-', log=False)
fig1.savefig(fig_dir + "profiles_sol.png")

# This function plots on top of the previous plot the profiles of r, l, P, T from the initial guess' integration
fig1, ax1 = sol_plotter(outward_guess, inward_guess, fig=fig1, ax=ax1, linestyle='dotted', color='grey', log=False, label='1st guess')
fig1.savefig(fig_dir + "profiles_sol_first_guess.png")

# This function plots the profiles of rho, kappa, the dels, and the epsilons
fig2, ax2 = plotter2(sol_dict, vis='default')
fig2.savefig(fig_dir + "profiles2_sol.png")



# Save the results as a csv file
sol_dict_for_csv = make_sol_dict(outward, inward, units_for_csv=True)
sol_df = pd.DataFrame.from_dict(sol_dict_for_csv)
profiles_dir = "./profiles/"
if not os.path.exists(profiles_dir):
    os.makedirs(profiles_dir)
    print("profiles directory created")
sol_df.to_csv(profiles_dir + 'profile_' + str(M/Ms) + ".csv")




# Open the MESA ZAMS results
MESA_dir = "./MESA_files/"
profile_name = "1.1M_at_ZAMS_last_profile.data"
history_name = "1.1M_at_ZAMS_history.data"
profile_path = MESA_dir + profile_name
history_path = MESA_dir + history_name

MESA_history = mesa_reader.MesaData(history_path)
MESA_profile = mesa_reader.MesaData(profile_path)

# Print the MESA results
print('#############################')
print('MESA parameters for the star:')
print('R_* = ', 10**MESA_history.log_R[-1] * Rs)
print('    = ', 10**MESA_history.log_R[-1], 'R_sun')
print('L_* = ', 10**MESA_history.log_L[-1] * Ls)
print('    = ', 10**MESA_history.log_L[-1], 'L_sun')
print('P_c = ', 10**MESA_history.log_cntr_P[-1])
print('T_c = ', 10**MESA_history.log_cntr_T[-1])
print('#############################')


print('########################################')
print('MESA additional parameters for the star:')
print('g = ', 10**MESA_history.log_g[-1])
print('T_eff = ', 10**MESA_history.log_Teff[-1])
print('########################################')

# Plot the MESA profiles of r, P and T
fig5, ax5 = MESA_plotter3(sol_dict, MESA_profile)
fig5.savefig(fig_dir + "profiles_MESA.png")

# Plot the MESA profiles of the epsilons
fig6, ax6 = MESA_eps_plotter(sol_dict, MESA_profile)
fig6.savefig(fig_dir + "MESA_eps.png")

#fig7, ax7 = MESA_mf_plotter(MESA_profile)
#fig7.savefig(fig_dir + "MESA_mf.png")


