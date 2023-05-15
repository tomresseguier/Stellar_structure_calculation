#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 16:18:45 2023

@author: tom
"""


import numpy as np
from matplotlib import pyplot as plt


def MESA_profile_reader(profile_path) :
    profile_file = open(profile_path, 'r')
    lines = profile_file.readlines()
    
    
    start_index = 6
    N_steps = int(lines[-1].split()[0])
    
    results_dict = {}
    columns = lines[5].split()
    for column in columns :
        results_dict[column] = np.zeros(N_steps)
    
    for i, line in enumerate(lines[start_index:]) :
        for j, column in enumerate(columns) :
            results_dict[column][i] = line.split()[j]
    return results_dict
