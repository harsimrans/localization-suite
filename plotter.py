#!/usr/bin/env python

import matplotlib.pyplot as plt
import math
import pandas as pd

import sys
from argparse import ArgumentParser

def cdist(x1, y1, x2, y2):
    """ 
        return the eucledian distance between two points
    """
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def plot(lat, lon, rss, trans_x, trans_y):
    """ 
        Plots the current snapshot of the environment
        The id, location and the observed RSS value
    """

    message = "lenths of lists latitude longitude and rss must be equal"
    assert(len(lat) == len(lon) and len(lat) == len(rss)), message

    fig, ax = plt.subplots()
    ax.scatter(lat, lon, color="blue")

    ax.scatter(lat, lon)

    for i, txt in enumerate(rss):
        label = str(i+1) + ' ' + str(round(cdist(lat[i], lon[i], trans_x,\
             trans_y), 2)) + " " + str(round(float(txt), 2))
        ax.annotate(label, (lat[i], lon[i]))

    plt.xlabel("X-coordinate (m)")
    plt.ylabel("Y-coordinate (m)")
    plt.title("RSS values: (id, distance from transmitter, observed rss)")
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("-n", "--ntrans", dest="ntrans",\
         help="takes the transmitter number", required=True, type=int)
    
    args, other_args = parser.parse_known_args()

    NUMBER = args.ntrans - 1 # zero based indexing, for user 1 based 
    
    # read the files
    df = pd.read_csv("loc_val.csv", header=None)
    df2 = pd.read_csv("rss_val.csv", header=None)
    
    # the lat long and rss
    lat = df[0]
    lon = df[1]
    rss = df2[NUMBER]

    # transmitter lat lon
    trans_x = df[0][NUMBER]
    trans_y = df[1][NUMBER]

    plot(lat, lon, rss, trans_x, trans_y)


if __name__=='__main__':
    main()