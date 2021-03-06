#!/usr/bin/env python

import math, sys, random
import numpy as np
import pandas as pd
from multi_localize import *

from sklearn.cluster import *
from sklearn.metrics import silhouette_samples, silhouette_score

import sys
from argparse import ArgumentParser

RADIUS = 1000.0 # the radius in meters around which to consider 

def calculate_grid_centers(x_max, y_max, x_min, y_min, delta):
    '''
        calculates and return the centers of the grid voxels in the area
        defined by x_max, y_max, x_min, y_min
        x_min, x_max: min and max value of x coordinate
        y_min, y_max: min and max value of y coordinate
        delta: distance between the center of the voxels

    '''
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

def tweak_rss_powers(loc, power, max_mag=0.0, vary_privacy=True):
    ''' 
        for each location return a false/new location and the corresponding 
        power calculated at the false/new location
        loc: list of location of sensors, each element (x, y) pair
        max_mag: is the maximum magnitude of noise to define range of noise 
                    selection as (0, max_mag).
        vary_privacy: if to consider each user might have varying privacy level
                    Mimics this by randomly choosing max_noise for each user as
                    max_noise = random.uniform(0, max_mag)  
    '''
    PATH_LOSS_EXPONENT = 2.0
    new_loc = []
    new_power = []


    for i in range(len(loc)):
        
        if vary_privacy:
            max_noise = random.uniform(0, max_mag)
        else:
            max_noise = max_mag

        # CHOOSE the point randomly
        lat = loc[i][0] + random.uniform(-1*max_noise, max_noise)
        lon = loc[i][1] + random.uniform(-1*max_noise, max_noise)
        
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
                new_pow_avg += (100000.0/dist)**2 * power[j]
                total += (100000.0/dist) ** 2 

        new_pow_avg /= total    
        
        new_power.append(new_pow_avg)
        new_loc.append((lat, lon))
    return new_loc, new_power


def cdist(loc1, loc2):
    return (loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2

def get_score(false_loc, prox):
    dist = 0.0
    for l in prox:
        dist += cdist(false_loc, l)
    return dist

def proximity(loc, loc_list, num):
    sorted_loc = sorted (loc_list, key = lambda x: (x[0] - loc[0])**2 + (x[1] - loc[1])**2)
    return sorted_loc[:num]

def return_best_false_location(false_locs, locs):
    scores = []
    for f in false_locs:
        k = 6
        # pick the top k locations based on proxity
        prox = proximity(f, locs, k)
        scores.append(get_score(f, locs))
    return false_locs[np.argmax(scores)][0], false_locs[np.argmax(scores)][1]

def list_to_freq_dict(lis):
    d = {}
    for l in lis:
        if l in d:
            d[l] += 1
        else:
            d[l] = 1
    return d


def tweak_rss_powers_using_k(loc, power, max_mag=0.0, k=3, vary_privacy=True):
    ''' 
        for each location return a false/new location and the corresponding 
        power calculated at the false/new location
        loc: list of location of sensors, each element (x, y) pair
        max_mag: is the maximum magnitude of noise to define range of noise 
                    selection as (0, max_mag).
        vary_privacy: if to consider each user might have varying privacy level
                    Mimics this by randomly choosing max_noise for each user as
                    max_noise = random.uniform(0, max_mag)  
    '''

    PATH_LOSS_EXPONENT = 2.0
    new_loc = []
    new_power = []

    ## clustering code 
    X = np.array(loc)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    freq = list_to_freq_dict(kmeans.labels_)

    for i in range(len(loc)):
        
        if vary_privacy:
            max_noise = random.uniform(0, max_mag)
        else:
            max_noise = max_mag

        # CHOOSE the point randomly
        lat = loc[i][0] + random.uniform(-1*max_noise, max_noise)
        lon = loc[i][1] + random.uniform(-1*max_noise, max_noise)
        
        NUM_FALSE = 3
        

        # generate NUM_FALSE Locations
        false_locations = []
        trials = 100
        while len(false_locations) < 3 and trials < 100:
            lat = loc[i][0] + random.uniform(-1*max_noise, max_noise)
            lon = loc[i][1] + random.uniform(-1*max_noise, max_noise)
            #print "Cluster predict: ", kmeans.predict([[lat, lon]])
            if freq[kmeans.predict([[lat, lon]])[0]] > 0:
                false_locations.append((lat, lon))
            trials -= 1
        
        # check if false locations empty
        while len(false_locations) == 0:
            lat = loc[i][0] + random.uniform(-1*max_noise, max_noise)
            lon = loc[i][1] + random.uniform(-1*max_noise, max_noise)
            freq[kmeans.predict([[lat, lon]])[0]] -= 1
            false_locations.append((lat, lon))

        lat, lon = return_best_false_location(false_locations, loc)
        freq[kmeans.predict([[lat, lon]])[0]] -= 1

        sorted_loc, sorted_pow = zip(*sorted(zip(loc, power), key=lambda x: get_distance(lat, lon, x[0][0], x[0][1])))
        
        #calculate the new power weighted RSS average
        total = 0.0
        new_pow_avg = 0.0
        for j in range(min(k, len(loc))):
            dist = get_distance(lat, lon, sorted_loc[j][0], sorted_loc[j][1])
            if dist == 0.0:
                new_pow_avg = sorted_pow[j]
                total = 1.0
                break
            else:   
                new_pow_avg += (100000.0/dist)**2 * sorted_pow[j]
                total += (100000.0/dist) ** 2 

        new_pow_avg /= total    
        
        new_power.append(new_pow_avg)
        new_loc.append((lat, lon))
    return new_loc, new_power

def change_location(receivers_g, noi):
    new_loc = []
    for rec in receivers_g:
        lat = rec[0]
        lon = rec[1]
        lat += random.uniform(-1*noi, noi)
        lon += random.uniform(-1*noi, noi)
        new_loc.append((lat, lon))
    return new_loc

def experiment(gs, ge, nlist, sample, iterations, vary_privacy, change_rss):

    # read the data 
    rss = pd.read_csv("rss_val.csv", header=None)
    loc = pd.read_csv("loc_val.csv", header=None)

    grid_centers = calculate_grid_centers(10, 13, -5, -5, 1.0)

    # possible values of group size to check for
    group_sizes = range(gs, ge)
    
    #noise_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    noise_list = nlist
    
    for noi in noise_list:
        print "--> adding noise: ", noi
        for g in group_sizes:
            error_list = []
            for i in range(44):
                # repeat n number of times
                NUM = iterations
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
                        dist = (loc[0][j] - loc[0][index])**2 + (loc[1][j] -\
                             loc[1][index])**2 
                        if dist <= (RADIUS ** 2):
                            receivers.append((loc[0][j], loc[1][j]))
                            powers_watt.append(10 ** (rss[i][j] / 10))
                            powers.append(rss[i][j])
                
                while NUM:
                    receivers_g = []
                    powers_g = []

                    if sample == 'r': # random sampling
                        indexes = random.sample(range(0, len(receivers)), g)
                        for ind in indexes:
                            receivers_g.append(receivers[ind])
                            powers_g.append(powers_watt[ind])
                    elif sample == 'p': # proximity based sampling
                        
                        # pick the maxima 
                        rss_max = -1000.0 # resonable value, below this the \
                            #signal won't make any sense
                        index = None
                        for j in range(0, 43):
                                if powers_watt[j] > rss_max:
                                    index = j
                                    rss_max = powers_watt[j] 

                        # pick group size around that receiver
                        l = range(0, 43)
                        l = sorted(l, key=lambda x: math.sqrt((receivers[x][0] \
                            - receivers[index][0])**2 + (receivers[x][1] -\
                                 receivers[index][1]) **2))
                        
                        receivers_g = [receivers[x] for x in l[0:g]]
                        powers_g = [powers_watt[x] for x in l[0:g]]
                        lats = []
                        lons = []
                        for j in range(len(receivers_g)):
                            lats.append(receivers_g[j][0])
                            lons.append(receivers_g[j][1])
                    
                    if change_rss:
                        #receivers_g, powers_g = tweak_rss_powers(receivers_g,\
                        #                     powers_g, noi, vary_privacy)
                        receivers_g, powers_g = tweak_rss_powers_using_k(receivers_g,\
                                              powers_g, noi, 5, vary_privacy)
                    else:
                        receivers_g = change_location(receivers_g, noi)
                    # locate the transmitter
                    x_cap = localize(receivers_g, powers_g, grid_centers)

                    # pick top n
                    top_n = x_cap.argsort()[-1:][::-1]
                    error = 1000.0
                    best = 0
                    for n in top_n:
                        if get_distance(grid_centers[n][0], grid_centers[n][1],\
                                         loc[0][i], loc[1][i]) < error:
                            best = n
                            error = get_distance(grid_centers[n][0], \
                                    grid_centers[n][1], loc[0][i], loc[1][i])
                    
                    errors.append(error)
                    NUM -= 1
                error_list.append(np.mean(errors))
            print "    * error for group size: ", g, np.mean(error_list)


def main():
    parser = ArgumentParser()
    parser.add_argument("-g", "--grange", dest="ngroup", nargs='+', \
            required=True, type=int,  help="takes the range of group size")
    parser.add_argument("-n", "--nrange", dest="nrange", nargs='+', \
            default=[14.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],\
            type=float, help="takes the range of noise")
    parser.add_argument("-i", "--iter", dest="niters", default=1, type=int,\
            help="number of iterations to run each group size for")
    parser.add_argument("-s", "--sample", dest="sample", action="store", \
            choices=['r','p'], default='r', type=str, \
            help="how to sample: r (random) [default] and p (proximity)")
    parser.add_argument("-r", "--radius", dest="radius", default=1000.0,\
            type=float,  help="radius around local maxima to consider (default: 1000m)")
    parser.add_argument("-v", "--vpriv", dest="vpriv", action="store_true",\
            help="different users have different levels of privacy settings")
    parser.add_argument("-c", "--changerss", dest="crss", action="store_true",\
            help="whether to change the rss value or not")


    args, other_args = parser.parse_known_args()

    global RADIUS
    RADIUS = args.radius
    experiment(args.ngroup[0], args.ngroup[1]+1, args.nrange, args.sample, \
        args.niters, args.vpriv, args.crss)



if __name__=='__main__':
    main()
