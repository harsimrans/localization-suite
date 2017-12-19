#!/usr/bin/env python

import math, sys, random
import numpy as np
import pandas as pd
from multi_localize import *

RADIUS = 4.0 # the radius in meters around which to consider 

def read_file(filename, second):
    '''reads the receiver data for the particular second
       returns a dictionary of data with receiver id as key'''
    f = open(filename, 'r')
    data_dict = {}
    for line in f.readlines():
      data = line.strip("\n").split(" ")
      sec = int(data[0]) / 1000
      #if sec == second:
      # data_dict[data[1]] = data[2:]
      if sec not in data_dict:
        data_dict[sec] = [(data[1], data[2:])]
      else:
        data_dict[sec].append((data[1], data[2:]))
            
    f.close()   
    return data_dict, sec

def calculate_grid_centers(x_max, y_max, x_min, y_min, delta):
    '''calculates and return the centers of the grid voxels'''
    x_centers = []
    y_centers = []
    while x_min <= x_max:
        x_centers.append(x_min)
        x_min += delta
    while y_min <= y_max:
        y_centers.append(y_min)
        y_min += delta

    # form the centers
    centers = []
    for x in x_centers:
      for y in y_centers:
        centers.append((x, y))
    return centers

def main():
    # read the file
    rss = pd.read_csv("rss_val.csv", header=None)
    loc = pd.read_csv("loc_val.csv", header=None)

    #print "Std dev: ", 7
    #GRID_SIZE = [2.5, 2.0, 1.5, 1.0, 0.5, 0.4, 0.3]
    GRID_SIZE = [1.0]
    #NOISE = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    NOISE = [0.0]

    for g in GRID_SIZE:
        print "Grid Size: ", g
        # get the grid centers
        grid_centers = calculate_grid_centers(10, 13, -5, -5, g)

        for n in NOISE:
            error_list = []
            print "NOISE: ", n
            for i in range(44):
                # get the maxima

                rss_temp = rss[:]
                rss_temp = rss_temp[i]
                rss_temp.pop(i)
                index = rss_temp.argmax()

                # pick receivers around that location
                receivers = []
                powers = []

                for j in range(44):
                    if i != j:
                        dist = (loc[0][index] - loc[0][j])**2 \
                            + (loc[1][index] - loc[1][j])**2 
                        if dist <= (RADIUS ** 2):
                            receivers.append((loc[0][j], loc[1][j]))
                            powers.append(10 ** (rss[i][j] / 10))

                x_cap = localize(receivers, powers, grid_centers)
                max_index = np.argmax(x_cap)
                location_transmitter = grid_centers[max_index]
                print "location of the transmitter at", i, " : ",\
                    location_transmitter, " actual: ", loc[0][i], loc[1][i],\
                    " error: ", get_distance(location_transmitter[0], \
                    location_transmitter[1], loc[0][i], loc[1][i]),\
                    " receivers used: ", len(receivers)

if __name__=='__main__':
    main()


