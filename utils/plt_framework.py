#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:09:43 2023

@author: Tom
"""


from matplotlib import pyplot as plt

def plt_framework() :
    font={'size':16, 'family':'serif'}
    plt.rc('font',**font)
    plt.rcParams['ytick.labelsize']='large'
    plt.rcParams['xtick.labelsize']='large'
    plt.rcParams['axes.labelsize']=24
    
    plt.rcParams['figure.figsize']=(12,9)
    
    plt.rcParams['xtick.minor.visible']='True'
    plt.rcParams['ytick.minor.visible']='True'
    plt.rcParams['figure.dpi']=60
    
    plt.rcParams['xtick.top']=True
    plt.rcParams['ytick.right']=True
    
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    
    plt.rcParams['xtick.major.top']=True
    plt.rcParams['xtick.minor.top']=True
    plt.rcParams['ytick.major.right']=True
    plt.rcParams['ytick.minor.right']=True
    
    plt.rcParams['xtick.major.size']=10
    plt.rcParams['xtick.minor.size']=6
    plt.rcParams['ytick.major.size']=10
    plt.rcParams['ytick.minor.size']=6
    
    plt.rcParams['xtick.major.width']=1.75
    plt.rcParams['xtick.minor.width']=1.2
    plt.rcParams['ytick.major.width']=1.75
    plt.rcParams['ytick.minor.width']=1.2
    
    plt.rcParams['savefig.bbox']='tight'
    plt.rcParams['savefig.format']='pdf'
    plt.rcParams['savefig.dpi']=300
    
    #plt.rcParams['text.usetex']=True
