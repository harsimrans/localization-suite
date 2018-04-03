from __future__ import print_function
from localize.localize import *
import os
import numpy as np
from matplotlib import pyplot as plt
import theano
import theano.tensor as T
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import math
import scipy
from scipy.interpolate import griddata
from scipy.optimize import linear_sum_assignment


SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



### Notations
# true_... = the ground truth
# ..._false = the falsely reported values
# ...true = the adversary's guess of true locations
#

def estimate_rss(receivers, x, y):
	rss_list = []
	for i in range(len(x)):
		rss = 0.0
		total_weight = 0.0
		for j in range(len(receivers)):
			dist = edist(receivers[j][0], receivers[j][1], x[i], y[i])
			if dist  == 0.0:
				rss = receivers[j][2]
				total_weight = 1.0
				break
			rss += (1/(dist **2)) * receivers[j][2]
			total_weight +=  1/(dist **2)
		rss_list.append(rss/total_weight)
	return rss_list


def ospa(x_true, y_true, true_x, true_y):
	# build a cost matrix
	C = [[None]*len(x_true) for _ in range(len(x_true))]
	for i in range(len(x_true)):
		for j in range(len(x_true)):
			C[i][j] = edist(x_true[i], y_true[i], true_x[j], true_y[j])
	C = np.array(C)
	row_ind, col_ind = linear_sum_assignment(C)
	#print (C[row_ind, col_ind])
	return C[row_ind, col_ind].sum() / len(x_true), C[row_ind, col_ind]


def MLE_attack(receivers, x_false, y_false, xmin, xmax, ymin, ymax):
	# make gueses for true locations and RSS
	x_true = [random.uniform(xmin, xmax) for i in range(len(receivers))]
	y_true = [random.uniform(ymin, ymax) for i in range(len(receivers))]
	rss_true = [random.uniform(-60, 1) for i in range(len(receivers))] # Random Initialization 

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
		d = d ** -1
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



NUM_REC = 44
TRANSMITTER_NUMBER = 26
loc, rss = read_dataset()
recv_list, trans_list = get_receiver_snapshots(loc, rss, NUM_REC)

receivers = recv_list[TRANSMITTER_NUMBER]
transmitter_loc = trans_list[TRANSMITTER_NUMBER]


ITERATIONS = 100
GROUP = 15
group_sizes = [10,15,25,30,35,40]

distances_lists = []
random_distances_lists = []
for k in range(len(group_sizes)):
	ospa_error_list = []
	
	distances_list = []
	random_distances_list = []
	for i in range(ITERATIONS):
		receivers_sampled = random.sample(receivers,group_sizes[k])

		# extract true locations
		true_x = [x[0] for x in receivers_sampled]
		true_y = [x[1] for x in receivers_sampled]
		true_rss = [x[2] for x in receivers_sampled]

		x_coor = []
		y_coor = []
		for r in receivers_sampled:
			x_coor.append(r[0])
			y_coor.append(r[1])

		xmax = max(x_coor)
		xmin = min(x_coor)
		ymax = max(y_coor)
		ymin = min(y_coor)

		# set up false locations to report
		x_false = [random.uniform(xmin, xmax) for m in range(len(receivers_sampled))]
		y_false = [random.uniform(ymin, ymax) for m in range(len(receivers_sampled))]
		# estimate the RSS at the false location 
		rss_false = estimate_rss(receivers_sampled, x_false, y_false)


		x_true, y_true, rss_true, loss_list = MLE_attack(receivers_sampled, x_false, y_false,  xmin, xmax, ymin, ymax)

		ospa_err , _distances = ospa(x_true, y_true, true_x, true_y)
		for d in _distances:
			distances_list.append(d)

		#print ("OSPA error: ", ospa_err)
		ospa_error_list.append(ospa_err)

		x_random = [random.uniform(xmin, xmax) for m in range(len(receivers_sampled))]
		y_random = [random.uniform(ymin, ymax) for m in range(len(receivers_sampled))]
		random_ospa_error, _random_distances = ospa(x_random, y_random, true_x, true_y)
		for d in _random_distances:
			random_distances_list.append(d)
	#print("list: ", ospa_error_list)
	print ("Average OSPA/matching error: ", np.mean(ospa_error_list), np.std(ospa_error_list))
	distances_lists.append(distances_list)
	random_distances_lists.append(random_distances_list)

inverse_attack_distances = pd.DataFrame(distances_lists)
random_guess_distances = pd.DataFrame(random_distances_lists)

inverse_attack_distances.to_csv('inverse_attack_distances.csv', index=False, header=False)
random_guess_distances.to_csv('random_guess_distances.csv', index=False, header=False)


plt.figure()
for j in range(len(group_sizes)):
	plt.plot(sorted(distances_lists[j]), np.linspace(0,1,len(distances_lists[j])), label=str(group_sizes[j]))
plt.title("Distance of matching for Inverse Attack")
plt.xlabel("Distance (m)")
plt.ylabel("Cummulative fraction")
plt.legend(loc='lower right', title="Group Size")
plt.xlim(0, 13)

plt.figure()
for j in range(len(group_sizes)):
	plt.plot(sorted(random_distances_lists[j]), np.linspace(0,1,len(random_distances_lists[j])), label=str(group_sizes[j]))
plt.title("Distance of matching for Random Guess")
plt.xlabel("Distance (m)")
plt.ylabel("Cummulative fraction")
plt.legend(loc='lower right'. title="Group Size")
plt.xlim(0, 13)

plt.show()

