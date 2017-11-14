import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import math
import scipy


######### HELPER FUNCTIONS ##################

def edist(x1, y1, x2, y2):
	"""
		calculates the eucledian distance
	"""
	return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_grid_centers(x_max, y_max, x_min, y_min, delta):
    """
        calculates and return the centers of the grid voxels in the area
        defined by x_max, y_max, x_min, y_min
        x_min, x_max: min and max value of x coordinate
        y_min, y_max: min and max value of y coordinate
        delta: distance between the center of the voxels

        :type x_max, y_max, x_min, y_min: float/int
        :type delta = float/int
        :rtype: List[(float, float)] / List[(int, int)] 
    """

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

########### END OF HELPER FUNCTIONS ###############

########### PLOTTING FUNCTIONS ####################

def plot_with_trans(lat, lon, rss, trans_x, trans_y, filename=None):
    """ 
        Plots the current snapshot of the environment
        The id, location and the observed RSS value

        Shows the plot if filename not present
        else saves at the given file location

        :type lat, lon, rss: List[float/int]
        :type transx = float/int
        :type transy = float/int
        
        :None 
    """

    message = "lenths of lists latitude longitude and rss must be equal"
    assert(len(lat) == len(lon) and len(lat) == len(rss)), message

    fig, ax = plt.subplots()
    ax.scatter(lat, lon, color="blue")

    for i, txt in enumerate(rss):
        label = str(i+1) + ' ' + str(round(edist(lat[i], lon[i], trans_x,\
             trans_y), 2)) + " " + str(round(float(txt), 2))
        ax.annotate(label, (lat[i], lon[i]))

    plt.xlabel("X-coordinate (m)")
    plt.ylabel("Y-coordinate (m)")
    plt.title("RSS values: (id, distance from transmitter, observed rss)")
    if save_file:
    	plt.savefig(filename)
    else:
    	plt.show()

def plot_recievers(lat, lon, rss, filename=None):
    """ 
        Plots the current snapshot of the environment
        The id, location and the observed RSS value
    """

    message = "lenths of lists latitude longitude and rss must be equal"
    assert(len(lat) == len(lon) and len(lat) == len(rss)), message

    fig, ax = plt.subplots()
    ax.scatter(lat, lon, color="blue")

    for i, txt in enumerate(rss):
        label = str(i+1) + ' ' + str(round(float(txt), 2))
        print label
        ax.annotate(label, (lat[i], lon[i]))

    plt.xlabel("X-coordinate (m)")
    plt.ylabel("Y-coordinate (m)")
    plt.title("RSS values: (id, distance from transmitter, observed rss)")
    
    if filename: 
        plt.savefig(filename)
    else: 
        plt.show()



########### END OF PLOTTING FUNCTIONS ##############


###### FUNCTIONS FOR LOCALIZATION #################
def get_weight(x1, y1, x2, y2):
	"""
		Computes the weight / influence fator between two locations
		represented by (x1, y1) and (x2, y2)
		
		:type x1, x2, x3, x4: float
        :rtype: float
	"""

	# define constants
	MINPL = 1.5
	PATHLOSS = 2.3

	dist = edist(x1, y1, x2, y2)

	if dist > MINPL:
	  return dist ** (-1 * PATHLOSS)
	else:
	  return MINPL ** (-1 * PATHLOSS)

def compute_weight_matrix(receivers, voxels):
	""" 
		calculates the weight matrix which captures 
		the degree of influence on each receiver for every
		transmitter location 

		:type receivers: List[(float, float)]
        :type voxels: List[(float, float)]

        :rtype: numpy array of len(receivers) x len(voxels)

	"""
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
	"""
		gets the covariance matrix for the least squares regularization
	"""
	C = np.empty(shape=(len(voxels), len(voxels)))
	counter = 0
	for loc1 in voxels:
		c = []
		for loc2 in voxels:
			c.append((sigma2 * math.e**(-1 * edist(float(loc1[0]), float(loc1[1]), float(loc2[0]), float(loc2[1])) / delta)))
		C[counter] = c 
		counter += 1
	return C

def localize_prob(receivers, powers, grid_centers, prob=True):
	"""
		Give a list of locations on a grid.
		Returns the likehood of transmitter being
		at each of the given list of locations.

		:type receivers: List[(float, float)]
        :type powers: List[float]
        :type grid_centers: List[(float, float)]
        :type prob: Bool
        :rtype: List[float]
	"""
	W = compute_weight_matrix(receivers, grid_centers)
	C = calculate_covariance_matrix(grid_centers)
	pi = np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(W), W), np.linalg.inv(C))), np.transpose(W))
	x_cap = np.matmul(pi, np.array(powers))

	# return the measure of some likelihood of each location
	return x_cap

def localize(receivers, powers, grid_centers, top_n=1):
	"""
		Give a list of locations on a grid.
		Returns the list of top n locations candidates
		for transmitter location.

		:type receivers: List[(float, float)]
        :type powers: List[float]
        :type grid_centers: List[(float, float)]
        :type top_n: int

        :rtype: List[(float, float)]
	"""
	n = int(n)

	W = compute_weight_matrix(receivers, grid_centers)
	C = calculate_covariance_matrix(grid_centers)
	pi = np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(W), W), np.linalg.inv(C))), np.transpose(W))
	x_cap = np.matmul(pi, np.array(powers))

	top_n = x_cap.argsort()[-1:][::-n] # top n indexes

	trans_candidates = []
	for n in top_n:
		trans_candidates.append((grid_centers[n][0], grid_centers[n][1]))

	return trans_candidates

################## END OF FUNCTIONS FOR LOCALIZATION ########################

