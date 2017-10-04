#!/usr/bin/env python

import math, sys, random
import numpy as np
import pandas as pd
from multi_localize import *
#import pylab

RADIUS = 100.0 # the radius in meters around which to consider 

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
def tweak_rss_powers(loc, power, max_mag=0.0):
    ''' mag is the magnitude of noise '''
    PATH_LOSS_EXPONENT = 2.0
    new_loc = []
    new_power = []
    
    
    for i in range(len(loc)):
        #mag = random.uniform(0, 5)
        mag = random.uniform(0, max_mag)
        # CHOOSE the point randomly
        lat = loc[i][0] + random.uniform(-1*mag, mag)
        lon = loc[i][1] + random.uniform(-1*mag, mag)
        
        #calculate the new power weighted RSS average
        total = 0.0
        new_pow_avg = 0.0
        for j in range(len(loc)):
            dist = get_distance(lat, lon, loc[j][0], loc[j][1])
            if dist == 0.0:
                new_pow_avg = power[j]
                total = 1.0
                break
            else:   
                new_pow_avg += (100000.0/dist)**4 * power[j]
                total += (100000.0/dist) ** 4. 
            #dist = (get_distance(lat, lon, t_loc[0], t_loc[1]) - get_distance(t_loc[0], t_loc[1], loc[j][0], loc[j][1])) ** 2
        new_pow_avg /= total    
        
        new_power.append(new_pow_avg)
        new_loc.append((lat, lon))
    return new_loc, new_power

def experiment():

    # read the data 
    rss = pd.read_csv("rss_val.csv", header=None)
    loc = pd.read_csv("loc_val.csv", header=None)

    grid_centers = calculate_grid_centers(10, 13, -5, -5, 1.0)

    # possible values of group size to check for
    group_sizes = range(1, 44)
    
    noise_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    noise_list = []
    print "check"
    for noi in noise_list:
        print "--> adding noise: ", noi
        for g in group_sizes:
            error_list = []
            for i in range(44):
                # repeat n number of times
                NUM = 1
                errors = []

                # prepare the list of receivers
                receivers = []
                powers = []
                powers_watt = []


                # pick the local maxima
                rss_temp = rss[:]
                rss_temp = rss_temp[i]
                rss_temp.pop(i)
                index = rss_temp.argmax()
                
                for j in range(44):
                            if i != j:
                                dist = (loc[0][j] - loc[0][index])**2 + (loc[1][j] - loc[1][index])**2 
                                if dist <= (RADIUS ** 2):
                                    receivers.append((loc[0][j], loc[1][j]))
                                    powers_watt.append(10 ** (rss[i][j] / 10))
                                    powers.append(rss[i][j])
                
                while NUM:
                    # randomly make groups of size g 
                    receivers_g = []
                    powers_g = []

                    indexes = random.sample(range(0, len(receivers)), g)
                    for ind in indexes:
                        receivers_g.append(receivers[ind])
                        powers_g.append(powers_watt[ind])

                    receivers_g, powers_g = tweak_rss_powers(receivers_g, powers_g, noi)
                    
                    # locate the transmitter
                    x_cap = localize(receivers_g, powers_g, grid_centers)

                    # pick top n
                    top_n = x_cap.argsort()[-1:][::-1]
                    error = 1000.0
                    best = 0
                    for n in top_n:
                        #bl = pylab.scatter(grid_centers[n][0], grid_centers[n][1], color="black", label="Best Picks")
                        #print "Distance from best: ", get_distance(grid_centers[top_n[0]][0], grid_centers[top_n[0]][1], grid_centers[n][0], grid_centers[n][1])
                        if get_distance(grid_centers[n][0], grid_centers[n][1], loc[0][i], loc[1][i]) < error:
                            best = n
                            error = get_distance(grid_centers[n][0], grid_centers[n][1], loc[0][i], loc[1][i])
                        #print "error at ", i, " ", error, " using best of ", len(top_n)
                    
                    errors.append(error)
                    NUM -= 1
                error_list.append(np.mean(errors))
            print "    * error for group size: ", g, np.mean(error_list)

if __name__=='__main__':
    experiment()