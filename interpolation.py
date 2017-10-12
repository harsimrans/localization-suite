import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import math
import scipy
from scipy.interpolate import griddata
import random

# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.griddata.html

loc = pd.read_csv("loc_val.csv", header=None)
rss = pd.read_csv("rss_val.csv", header=None)

t = []
for i in range(len(rss[24])):
	t.append((loc[0][i], loc[1][i], rss[24][i]))

# pick about 35 receivers
random.shuffle(t)
anchor = t[:35]
others = t[35:]

# prepare the interpolation
locations = []
rss_values = []
for a in anchor:
	locations.append((a[0], a[1]))
	rss_values.append(a[2])

pred_location = []
pred_rss = []
for o in others:
	pred_location.append((o[0], o[1]))
	pred_rss.append(o[2])

zi = scipy.interpolate.griddata(locations, rss_values, pred_location)

for i in range(len(pred_location)):
	print pred_location[i], zi[i], pred_rss[i]


print "try taking all points in the convex hull"
indexes = [4, 6, 7, 8, 15, 26, 28, 36, 37, 38, 40, 42]

locations = []
rss_values = []
pred_location = []
pred_rss = []

for i in range(44):
	print i, loc[0][i], loc[1][i], rss[24][i]
	if i in indexes:
		pred_location.append((loc[0][i], loc[1][i]))
		pred_rss.append(rss[24][i])
	else:
		locations.append((loc[0][i], loc[1][i]))
		rss_values.append(rss[24][i])

zi = scipy.interpolate.griddata(locations, rss_values, pred_location)

for i in range(len(pred_location)):
	print indexes[i], pred_location[i], zi[i], pred_rss[i]

