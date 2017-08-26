###########################
#
# This script localizes the transmitter
#
#
# 1. We need to find the local maxima at a given time instance
# 2. Select the receivers within a certain radius R of the local maxima
# 3. Select the voxels which cover these receivers completely
# 4. Apply the inverse model to find the voxel with higest power, this is the
#    location of the transmitter corresponding to first maxima


import numpy as np
import math
# define constants

MINPL = 1.5
PATHLOSS = 2.3


def get_distance(x1, y1, x2, y2):
	return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_weight(x1, y1, x2, y2):
	'''
		computes the weight / influence fator between two locations
	'''
	dist = get_distance(x1, y1, x2, y2)

	if dist > MINPL:
	  return dist ** (-1 * PATHLOSS)
	else:
	  return MINPL ** (-1 * PATHLOSS)

def compute_weight_matrix(receivers, voxels):
	''' calculates the weight matrix which captures 
		the degree of influence on each receiver for every
		transmitter location 
	'''
	W = np.empty(shape=(len(receivers), len(voxels)))
	counter = 0
	for loc_r in receivers:
		w = []
		for loc_v in voxels:
			w.append(get_weight(float(loc_r[0]), float(loc_r[1]), float(loc_v[0]), float(loc_v[1])))
		W[counter] = w
		counter += 1
	return W

def calculate_covariance_matrix(voxels, sigma2=0.5, delta=1.0):
	'''gets the covariance matrix for the least squares regularization'''
	C = np.empty(shape=(len(voxels), len(voxels)))
	counter = 0
	for loc1 in voxels:
		c = []
		for loc2 in voxels:
			c.append((sigma2 * math.e**(-1 * get_distance(float(loc1[0]), float(loc1[1]), float(loc2[0]), float(loc2[1])) / delta)))
		C[counter] = c 
		counter += 1
	return C

def localize(receivers, powers, grid_centers):
	W = compute_weight_matrix(receivers, grid_centers)
	#print W
	C = calculate_covariance_matrix(grid_centers)
	#print C
	pi = np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(W), W), np.linalg.inv(C))), np.transpose(W))
	x_cap = np.matmul(pi, np.array(powers))
	#print "length of x_cap", len(x_cap)

	# return the measure of some likelihood of each location 
	return x_cap

	## pick the max one
	#max_index = np.argmax(x_cap)
	#location_transmitter = grid_centers[max_index]
	#return location_transmitter


