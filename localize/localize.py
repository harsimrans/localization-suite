from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import math
import scipy
import os
import random
import theano
import theano.tensor as T
from scipy.optimize import linear_sum_assignment


def read_dataset():
    '''
        reads receiver location and rss values

    '''
    # read the data 
    RSS_FILE = os.path.join(os.path.dirname(__file__), 'rss.csv')
    LOC_FILE = os.path.join(os.path.dirname(__file__), 'loc.csv')
    
    #rss = pd.read_csv("rss_val.csv", header=None)
    #loc = pd.read_csv("loc_val.csv", header=None)
    rss = pd.read_csv(RSS_FILE, header=None)
    loc = pd.read_csv(LOC_FILE, header=None)
    return loc, rss

def get_receiver_snapshots(loc, rss, NUM):
    '''
        generates a list of receiver snapshots
    '''
    recv_list = []
    trans_list = []
    for i in range(NUM):
        
        # get the receivers
        receivers = []
        for j in range(NUM):
            if i != j:
                receivers.append([loc[0][j], loc[1][j], rss[i][j]])
        
        trans_list.append([loc[0][i], loc[1][i]])
        recv_list.append(receivers)
    return recv_list, trans_list



def add_location_noise(receivers, noi):
    noisy_receivers = []
    for rec in receivers:
        lat = rec[0] + random.uniform(-1*noi, noi)
        lon = rec[1] + random.uniform(-1*noi, noi)
        noisy_receivers.append([lat, lon, rec[2]])
    return noisy_receivers

def add_location_noise_vary_privacy(receivers, noi):
    noisy_receivers = []
    for rec in receivers:
        n = random.uniform(0, noi)
        lat = rec[0] + random.uniform(-1*n, n)
        lon = rec[1] + random.uniform(-1*n, n)
        noisy_receivers.append([lat, lon, receivers[2]])
    return noisy_receivers

def estimate_rss_powers(receivers, lat, lon):
    '''
       Given a set of receivers, estimate the rss at location (lat, lon)
    '''

    #calculate the new power weighted RSS average
    total = 0.0
    new_pow_avg = 0.0
    for or_rev in receivers:
        dist = edist(lat, lon, or_rev[0], or_rev[1])
        if dist == 0.0:
            new_pow_avg = or_rev[2]
            total = 1.0
            break
        else:   
            new_pow_avg += (100000.0/dist)**2 * or_rev[2]
            total += (100000.0/dist) ** 2 

    new_pow_avg /= total    
    
    return new_pow_avg


def false_location_estimate_rss(receivers, max_mag=0.0):
    dup_receivers = []
    
    for r in receivers:
        # CHOOSE the point randomly
        lat = r[0] + random.uniform(-1*max_mag, max_mag)
        lon = r[1] + random.uniform(-1*max_mag, max_mag)

        new_pow_avg = estimate_rss_powers(receivers, lat, lon)
        dup_receivers.append([lat, lon, new_pow_avg])
    return dup_receivers

def random_false_location(receivers, grid=(10, 13, -5, -5), num_sample=None):
    '''
       takes a list of receivers and grid dims
       grid dims: (xmax, ymax, xmin, ymin)
    '''
    xmin = grid[2]
    xmax = grid[0]
    ymin = grid[3]
    ymax = grid[1]

    false_receiver = []
    if num_sample == None:
        num_sample = len(receivers)

    for i in range(num_sample):
        # pick a location 
        lat = random.uniform(xmin, xmax)
        lon = random.uniform(ymin, ymax)

        # calculate the RSS
        power = estimate_rss_powers(receivers, lat, lon)

        false_receiver.append([lat, lon, power])
    return false_receiver


def select_subset(rev_loc, xmin, xmax, ymin, ymax):
    """
        Given a list of receiver location pairs along 
        with their RSS, picks only the receivers confined 
        in an area (xmin , xmax, ymin, ymax)
    """
    new_rl = []
    for r in rev_loc:
        if xmin <= r[0][0] <= xmax and ymin <= r[0][1] <= ymax:
            new_rl.append(r)
    return new_rl


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
    return np.asarray(centers)

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
        print(label)
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


def localize(receivers, grid_centers, top_n=1, rss_dbm=True):
    
    x_cap = localize_prob(receivers, grid_centers, rss_dbm)
    top_n = x_cap.argsort()[-1*top_n:][::-1] # top n indexes

    trans_candidates = []
    for n in top_n:
        trans_candidates.append((grid_centers[n][0], grid_centers[n][1]))

    return trans_candidates

def localize_prob(receivers, grid_centers, rss_dbm=True):
    # separate out the powers
    powers = np.asarray(receivers)[:, [2]]
    receivers = np.asarray(receivers)[:, [0,1]]
    if rss_dbm:
        powers = 10**(powers / 10)
    
    W = compute_weight_matrix(receivers, grid_centers)
    C = calculate_covariance_matrix(grid_centers)
    pi = np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(W), W), np.linalg.inv(C))), np.transpose(W))
    
    x_cap = np.matmul(pi, powers)
    x_cap = x_cap.reshape(x_cap.shape[0], )

    return x_cap


################## END OF FUNCTIONS FOR LOCALIZATION ########################

################## ADVERSARY ATTACK AND PRIVACY EVALUATION ##################


def inverse_attack(receivers, x_false, y_false, xmin, xmax, ymin, ymax, pathloss=2.0, rss_min=-60, rss_max=1):

    ### Notations
    # true_... = the ground truth
    # ..._false = the falsely reported values
    # ...true = the adversary's guess of true locations
    #

    # make gueses for true locations and RSS
    x_true = [random.uniform(xmin, xmax) for i in range(len(receivers))]
    y_true = [random.uniform(ymin, ymax) for i in range(len(receivers))]
    rss_true = [random.uniform(rss_min, rss_max) for i in range(len(receivers))] # Random Initialization 

    # fill RSS up with closest point in the estimate (Intellifgent RSS Initialization)
    rss_true = []
    for i in range(len(x_true)):
        distance = float('inf')
        index = None
        for j in range(len(x_false)):
            if edist(x_false[j], y_false[j], x_true[i], y_true[i]) < distance:
                index = j
                distance = edist(x_false[j], y_false[j], x_true[i], y_true[i])
        rss_true.append(rss_false[index]) 

    # Define elements in graph

    x = T.dmatrix('x') # the x coordinates of the true location to guess
    y = T.dmatrix('y') # the y coordinates of the true location to guess
    f = T.dmatrix('f') # the RSS values at the these locations

    vx = T.dscalar('vx') # falsely reported location x coordinate
    vy = T.dscalar('vy') # falsely reported location y coordinate

    p = T.dscalar('p') # adjusted RSS reported for false location reported


    def function(x, y, vx, vy, f):
        '''
            Takes Theno tensors as input and calculates the RSS at falsely reported 
            locations based on true location guesses
        '''
        d = (x - vx)**2 + (y - vy)**2 
        d = d ** (-1 * pathloss)
        predicted = T.sum((d / T.sum(d)) * f)
        return predicted
        

    def loss(x, y, vx, vy, f, p):
        '''
            Calculated the loss for single instance of falsely reported location
        '''
        return (function(x, y, vx, vy, f) - p)**2

    # set for partial gradients
    gx = T.grad(loss(x, y, vx, vy, f, p), x)
    gy = T.grad(loss(x, y, vx, vy, f, p), y)
    gf = T.grad(loss(x, y, vx, vy, f, p), f)

    # convert in Theano function for calculations
    f1 = theano.function([x,y,vx,vy,f,p], gx)
    f2 = theano.function([x,y,vx,vy,f,p], gy)
    f3 = theano.function([x,y,vx,vy,f,p], gf)

    # factor out the loss function to calculate values for plotting
    f_loss = theano.function([x,y,vx,vy,f,p], (function(x, y, vx, vy, f) - p)**2)


    #loss_function = theano.function([x,y,vx,vy,f,p], loss)

    loss_list = []
    EPOCH = 1
    for m in range(EPOCH):
        delta = 0.01 # incremental update size
        num_trials = 500 # number of iterations for MLE


        # loop through and find the gradient for all the vjs; update guesses and repeat
        
        counter = 1
        for k in range(num_trials):
            sumf1 = [0.0] * len(x_true)
            sumf2 = [0.0] * len(x_true)
            sumf3 = [0.0] * len(x_true)
            cumm_loss = 0.0
            for i in range(len(x_false)):
                #cumm_loss += math.sqrt(f_loss([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i]))
                cumm_loss +=f_loss([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i])
                sumf1 += f1([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i])
                sumf2 += f2([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i])
                sumf3 += f3([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i])
            
            #loss_list.append(cumm_loss/len(x_false))   
            loss_list.append(cumm_loss) 
            

            #print("loss for interation", k, loss([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i]))
            # update the true location and rss guesses
            for i in range(len(x_true)):
                x_true[i] -= delta * sumf1[0][i]
                y_true[i] -= delta * sumf2[0][i]
                rss_true[i] -= delta * sumf3[0][i]

            #delta = 0.01 / math.sqrt(counter)
            counter += 1
    return x_true, y_true, rss_true, loss_list



def min_cost_matching(x_true, y_true, true_x, true_y):
    '''
        matches locations in adversary's location guess 
        to the true location of the receivers based on
        minimum cost bipartite matching
    '''
    # build a cost matrix
    C = [[None]*len(x_true) for _ in range(len(x_true))]
    for i in range(len(x_true)):
        for j in range(len(x_true)):
            C[i][j] = edist(x_true[i], y_true[i], true_x[j], true_y[j])
    C = np.array(C)
    row_ind, col_ind = linear_sum_assignment(C)
    #print (C[row_ind, col_ind])
    return C[row_ind, col_ind].sum() / len(x_true), C[row_ind, col_ind]
