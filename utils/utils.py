#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 22:57:03 2023

@author: Tom
"""


import numpy as np


def opacity_reader(path, table_number=74) :
    opacity_tables = open(path, 'r')
    lines = opacity_tables.readlines()
    table_name = 'TABLE # ' + str(74)
    table_index = np.where([table_name in line for line in lines])[0][1]
    
    logR_range_str = lines[table_index + 4].split()[1:]
    logR_range = [float(num_str) for num_str in logR_range_str]
    
    logT_range_str = [lines[table_index + 6 + i].split()[0] for i in range(70)]
    logT_range = [float(num_str) for num_str in logT_range_str]
    
    table = np.zeros((len(logT_range), len(logR_range)))
    
    for i in range(len(table)) :
        line_str = lines[table_index + 6 + i].split()[1:]
        if len(line_str) < 19 :
            for j in range(19 - len(line_str)) :
                line_str.append('nan')
        table[i] = np.array([float(num_str) for num_str in line_str])
    
    return table, logR_range, logT_range


