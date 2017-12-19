#!/usr/bin/env python

from scipy.interpolate import Rbf
import pandas as pd
import numpy as np
import math

def get_estimated_rss(xr, yr, rssr, xe, ye, sv=0):
    ''' takes as input reference points in the form
        xr: x-coordinate of location
        yr: y-coordinate of location
        rssr: rss at the x, y coordinate
        xe: x-coordinate list to estimated
        ye: y-coordinate list of points to be estimated
        smooth: level of smoothening required

        returns: the estimated value of the RSS
    '''

    model = Rbf(xr, yr, rssr, smooth=sv)
    out = model(xe, ye)
    return out

def MSE(x, y):
    """
        takes as input two lists x, y (true, estimated)
        and calculates the Mean Squared Error
        
        returns: MSE of the two list x, y
    """

    assert(len(x) == len(y)), "The list sizes should be equal"
    # if the lists doesn't contain value, the error is zero
    if len(x) == 0:
        return 0.0
    error = 0.0
    total = 0
    for i in range(len(x)):
        error += (x[i] - y[i])**2
        total += 1
    return error/(total + 0.0)

def MAE(x, y):
    """
        takes as input two lists x, y (true, estimated)
        and calculates the Mean Absolute Error
        
        returns: Mean absolute of the two lists
    """

    assert(len(x) == len(y)), "The list sizes should be equal"
    # if the lists doesn't contain value, the error is zero
    if len(x) == 0:
        return 0.0
    error = 0.0
    for i in range(len(x)):
        error += math.fabs(x[i] - y[i])
    return error / len(x)

def experiment(loc, rss, txid=None):
    """
        Runs an experiment for random size of anchor nodes and
        different values of smoothening for RBF interpolation

        loc: list of location of the receivers
        rss: list of rss values observed the the receivers
    """

    if txid:
        print "For transmitter: ", txid
    
    # begin with atleat 25 % anchor nodes
    for i in range(len(loc)/4, len(loc)-2):

        # pick set of unique anchor nodes randomly
        idx = np.random.choice(np.arange(len(loc)), i+1, replace=False)
        
        # lists for anchor nodes
        loc_anchor_x = []
        loc_anchor_y = []
        rss_anchor = []
        
        # lists for nodes to predict and evaluate
        loc_pred_x = []
        loc_pred_y = []
        rss_pred = []

        # populate the lists
        for j in range(len(loc)):
            if j == txid:
                continue
            if j in idx:
                loc_anchor_x.append(loc[0][j])
                loc_anchor_y.append(loc[1][j])
                rss_anchor.append(rss[txid][j])
            else:
                loc_pred_x.append(loc[0][j])
                loc_pred_y.append(loc[1][j])
                rss_pred.append(rss[txid][j])

        # Smooth values to try for RBF kernel
        smooth_vals = [1,3,5,7,10]

        for smooth_val in smooth_vals:
            MAEs = []
            MSEs = []
            #print len(loc_anchor_x) ,len(loc_anchor_y), len(rss_anchor), \
            #   len(loc_pred_x), len(loc_pred_y)
            output = get_estimated_rss(loc_anchor_x, loc_anchor_y, rss_anchor, \
                loc_pred_x, loc_pred_y, smooth_val)
            MAEs.append(MAE(output, rss_pred))
            MSEs.append(MSE(output, rss_pred))
        print "\t MSE: {0:10.4f}    MAE: {1:10.4f}    "\
            "Num anchor nodes: {2:3d}".format(min(MSEs), min(MAEs),\
             len(loc_anchor_x))


def main():
    # read the files
    loc = pd.read_csv("loc_val.csv", header=None)
    rss = pd.read_csv("rss_val.csv", header=None)
    for i in range(44):
        experiment(loc, rss, i)

if __name__=='__main__':
    main()