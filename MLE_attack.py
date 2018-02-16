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
			rss += (1/(dist **2)) * receivers[j][2]
			total_weight +=  1/(dist **2)
		rss_list.append(rss/total_weight)
	return rss_list


NUM_REC = 44
TRANSMITTER_NUMBER = 26
loc, rss = read_dataset()
recv_list, trans_list = get_receiver_snapshots(loc, rss, NUM_REC)

receivers = recv_list[TRANSMITTER_NUMBER]
transmitter_loc = trans_list[TRANSMITTER_NUMBER]


# extract true locations
true_x = [x[0] for x in receivers]
true_y = [x[1] for x in receivers]
true_rss = [x[2] for x in receivers]

# ### Convert true RSS to Watts
# true_rss_watt = []
# for val in true_rss:
# 	true_rss_watt.append(10**(val/ 10))
# true_rss = true_rss_watt[:]

# #also do in the receiver list
# for i in range(len(receivers)):
# 	receivers[i][-1] = 10**(receivers[i][-1]/ 10)


print(true_x)
print(true_y)
print(true_rss)

# set up false locations to report
x_false = [random.uniform(-5, 10) for i in range(len(receivers))]
y_false = [random.uniform(-1, 15) for i in range(len(receivers))]
# estimate the RSS at the false location 
rss_false = estimate_rss(receivers, x_false, y_false)


# test the estimated RSS by localizing the transmitter (Gives you a rough idea if everything is in place till now)
grid_centers = calculate_grid_centers(10, 13, -5, -5, 1.0)
noisy_recv_list = []
for i in range(len(x_false)):
	noisy_recv_list.append([x_false[i], y_false[i], rss_false[i]])
transmitter_localized = localize(noisy_recv_list, grid_centers, rss_dbm=True)[0]

print("the error with localize: ", edist(transmitter_localized[0], transmitter_localized[1], transmitter_loc[0], transmitter_loc[1]))


# START MOUNTING THE MLE ATTACK


def MLE_attack(receivers, x_false, y_false):
	# make gueses for true locations and RSS
	x_true = [random.uniform(-5, 10) for i in range(len(receivers))]
	y_true = [random.uniform(-5, 15) for i in range(len(receivers))]
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


	EPOCH = 4
	for m in range(EPOCH):
		delta = 0.01 # incremental update size
		num_trials = 1000 # number of iterations for MLE


		# loop through and find the gradient for all the vjs; update guesses and repeat
		loss_list = []
		counter = 1
		for k in range(num_trials):
			sumf1 = [0.0] * len(x_true)
			sumf2 = [0.0] * len(x_true)
			sumf3 = [0.0] * len(x_true)
			cumm_loss = 0.0
			for i in range(len(x_false)):
				cumm_loss += math.sqrt(f_loss([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i]))
				sumf1 += f1([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i])
				sumf2 += f2([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i])
				sumf3 += f3([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i])
			
			loss_list.append(cumm_loss/len(x_false))	
			#print("loss for interation", k, loss([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i]))
			# update the true location and rss guesses
			for i in range(len(x_true)):
				x_true[i] -= delta * sumf1[0][i]
				y_true[i] -= delta * sumf2[0][i]
				rss_true[i] -= delta * sumf3[0][i]

			delta = 0.01 / math.sqrt(counter)
			counter += 1

		# fix the guesses close to the false locations

		for i in range(len(x_true)):
			min_index = None
			mdist = float('inf')

			# pick the nearest false reported location
			for j in range(len(x_false)):
				if edist(x_true[i], y_true[i], x_false[j], y_false[j]) < mdist:
					mdist = edist(x_true[i], y_true[i], x_false[j], y_false[j])
					min_index = j
			if mdist < 1.5: # close to some falsely reported location by 1.5m
				# initialize to random 
				x_true[i] = random.uniform(-5, 10)
				y_true[i] = random.uniform(-5, 15)
	return x_true, y_true, rss_true, loss_list


# # make gueses for true locations and RSS
# x_true = [random.uniform(-5, 10) for i in range(len(receivers))]
# y_true = [random.uniform(-5, 15) for i in range(len(receivers))]
# rss_true = [random.uniform(-60, 1) for i in range(len(receivers))] # Random Initialization 

# # fill RSS up with closest point in the estimate (Intellifgent RSS Initialization)
# rss_true = []
# for i in range(len(x_true)):
# 	distance = 1000.0
# 	index = None
# 	for j in range(len(x_false)):
# 		if edist(x_false[j], y_false[j], x_true[i], y_true[i]) < distance:
# 			index = j
# 			distance = edist(x_false[j], y_false[j], x_true[i], y_true[i])
# 	rss_true.append(rss_false[index]) 


# # This is an initialization for quick sanity check for the Theano graph for MLE gradient updates
# # With ground truth the loss should be almost 0 and should stay around that

# # x_true = true_x[:]
# # y_true = true_y[:]
# # rss_true = true_rss[:]


# # Define elements in graph

# x = T.dmatrix('x') # the x coordinates of the true location to guess
# y = T.dmatrix('y') # the y coordinates of the true location to guess
# f = T.dmatrix('f') # the RSS values at the these locations

# vx = T.dscalar('vx') # falsely reported location x coordinate
# vy = T.dscalar('vy') # falsely reported location y coordinate

# p = T.dscalar('p') # adjusted RSS reported for false location reported


# def function(x, y, vx, vy, f):
# 	'''
# 		Takes Theno tensors as input and calculates the RSS at falsely reported 
# 		locations based on true location guesses
# 	'''
# 	d = (x - vx)**2 + (y - vy)**2 
# 	d = d ** -1
# 	predicted = T.sum((d / T.sum(d)) * f)
# 	return predicted
	

# def loss(x, y, vx, vy, f, p):
# 	'''
# 		Calculated the loss for single instance of falsely reported location
# 	'''
# 	return (function(x, y, vx, vy, f) - p)**2

# # set for partial gradients
# gx = T.grad(loss(x, y, vx, vy, f, p), x)
# gy = T.grad(loss(x, y, vx, vy, f, p), y)
# gf = T.grad(loss(x, y, vx, vy, f, p), f)

# # convert in Theano function for calculations
# f1 = theano.function([x,y,vx,vy,f,p], gx)
# f2 = theano.function([x,y,vx,vy,f,p], gy)
# f3 = theano.function([x,y,vx,vy,f,p], gf)

# # factor out the loss function to calculate values for plotting
# f_loss = theano.function([x,y,vx,vy,f,p], (function(x, y, vx, vy, f) - p)**2)


# #loss_function = theano.function([x,y,vx,vy,f,p], loss)



# delta = 0.01 # incremental update size
# num_trials = 4000 # number of iterations for MLE


# # loop through and find the gradient for all the vjs; update guesses and repeat
# loss_list = []
# counter = 1
# for k in range(num_trials):
# 	sumf1 = [0.0] * len(x_true)
# 	sumf2 = [0.0] * len(x_true)
# 	sumf3 = [0.0] * len(x_true)
# 	cumm_loss = 0.0
# 	for i in range(len(x_false)):
# 		cumm_loss += math.sqrt(f_loss([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i]))
# 		sumf1 += f1([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i])
# 		sumf2 += f2([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i])
# 		sumf3 += f3([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i])
	
# 	loss_list.append(cumm_loss/len(x_false))	
# 	#print("loss for interation", k, loss([x_true], [y_true], x_false[i], y_false[i], [rss_true], rss_false[i]))
# 	# update the true location and rss guesses
# 	for i in range(len(x_true)):
# 		x_true[i] -= delta * sumf1[0][i]
# 		y_true[i] -= delta * sumf2[0][i]
# 		rss_true[i] -= delta * sumf3[0][i]

# 	delta = 0.01 / math.sqrt(counter)
# 	counter += 1


x_true, y_true, rss_true, loss_list = MLE_attack(receivers, x_false, y_false)

# Out of curiosity, let's test how well the guessed RSS field does
grid_centers = calculate_grid_centers(10, 13, -5, -5, 1.0)
noisy_recv_list = []
for i in range(len(x_true)):
	noisy_recv_list.append([x_true[i], y_true[i], rss_true[i]])

transmitter_localized = localize(noisy_recv_list, grid_centers, rss_dbm=True)[0]
print ("locating transmitter again: ", edist(transmitter_localized[0], transmitter_localized[1], transmitter_loc[0], transmitter_loc[1]))


# let's give adversary the best evaluation metric, he cannot do better
error_list = []
for i in range(len(x_true)):
	err = float('inf')
	index = None
	for j in range(len(x_true)):
		if edist(x_true[i], y_true[i], true_x[j], true_y[j]) < err:
			index = j

			err = edist(x_true[i], y_true[i], true_x[j], true_y[j])
	error_list.append(err)


print("Metric for adversary's guess: ", np.mean(error_list), min(error_list), max(error_list)) # some stats

# how rss values relate to localization error

# prepare a list of rss vs error
rss_error_list = []
for i in range(len(x_true)):
	rss_error_list.append([error_list[i], rss_true[i]])

print("RSS error list (with predicted RSS): ")
rss_error_list.sort(key=lambda x: x[-1])
print(rss_error_list)

plt.figure()
rss_error_list = np.array(rss_error_list)
plt.plot(rss_error_list[:, [0]], rss_error_list[:, [1]])
#plt.show()

rss_error_list = []
for i in range(len(x_true)):
	rss_error_list.append([error_list[i], rss_false[i]])

print("RSS error list (with false reported RSS): ")
sorted(rss_error_list, key=lambda x: x[-1])
print(rss_error_list)

# Time for visualization

# The loss function
plt.figure()
plt.plot(loss_list)
plt.title("Loss Function")
plt.xlabel("Iteration number")
plt.ylabel("Loss calculated on RSS dBm")


# how well do guesses true locations fair against the ground truth for true locations
#plt.figure()
fig, ax = plt.subplots()

ax.scatter(x_true, y_true, label="Guesses true locations (by adversary)")
ax.scatter(true_x, true_y, label="Actual true locations (ground truth)")
ax.scatter(x_false, y_false, label="false locations")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1))
plt.xlabel("X-coordinate (m)")
plt.ylabel("Y-coordinate (m)")
ax.grid(True)


# Lets the RSS field from falsely reported data to server
plt.figure()
print(len(loc))
loc = loc.drop(loc.index[[TRANSMITTER_NUMBER]])
print (len(loc), len(true_rss))
yi = np.linspace(int(min(loc[1])), int(max(loc[1]))-1 , 19*3)
xi = np.linspace(int(min(loc[0]))+1, int(max(loc[0])) , 16*3)
zi = griddata((x_false, y_false), rss_false, (xi[None,:], yi[:,None]), method='linear')

plt.contour(xi, yi, zi, colors='k')
plt.contourf(xi, yi, zi, cmap=plt.cm.jet)
plt.xlabel("X-coordinate (m)")
plt.ylabel("Y-coordinate (m)")
plt.title("Variation of RSS in an Area(from falsely reported locations)")
cb = plt.colorbar()
cb.set_label('RSS values in dBm')

# let's see the RSS field from the guesses made the adversary
plt.figure()
zi = griddata((x_true, y_true), rss_true, (xi[None,:], yi[:,None]), method='linear')
plt.contour(xi, yi, zi, colors='k')
plt.contourf(xi, yi, zi, cmap=plt.cm.jet)
plt.xlabel("X-coordinate (m)")
plt.ylabel("Y-coordinate (m)")
plt.title("Variation of RSS in an Area(from Adversary Guesses)")
cb = plt.colorbar()
cb.set_label('RSS values in dBm')

plt.show()



