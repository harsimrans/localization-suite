#!/usr/bin/env python

#
# It simply adds noise to the receiver locations
# using different methods.
# Also given an option to adjust RSS measurements
#
#

from __future__ import print_function
from localize.localize import *
import os

def calculate_localization_error(recv_list, trans_list, grid_centers, k, verbose=False):	
	NOISE_VAL = 10.0

	error = []
	
	for i in range(len(recv_list)):
		curr_recv_list = recv_list[i]
		
		# sample k receivers
		sampled_receivers = random.sample(curr_recv_list, k)
		
		# noisy_recv_list = add_location_noise(recv_list[i], NOISE_VAL)
		# noisy_recv_list = add_location_noise_vary_privacy(recv_list[i], NOISE_VAL)
		# noisy_recv_list = false_location_estimate_rss(sampled_receivers, NOISE_VAL)
		noisy_recv_list = random_false_location(recv_list[i], grid=(10, 13, -5, -5))
		tx_loc = localize(noisy_recv_list, grid_centers)[0]
		e = edist(tx_loc[0], tx_loc[1], trans_list[i][0], trans_list[i][1])
		error.append(e)

		if verbose:
			print("Error:", "{0:.2f}".format(e), "m")
	avg_error = np.mean(error)
	if verbose:
		print("Average Error: ", "{0:.2f}".format(avg_error), "m")
	return avg_error

def multirun():
	NUM_REC = 43

	# read the data
	loc, rss = read_dataset()
	recv_list, trans_list = get_receiver_snapshots(loc, rss, NUM_REC)
	grid_centers = calculate_grid_centers(10, 13, -5, -5, 1.0)

	NUM_TRIALS = 100
	
	for j in range(42, 43):
		error_list = []
		for i in range(NUM_TRIALS):
			e = calculate_localization_error(recv_list, trans_list, grid_centers, j)
			# print ("TRIAL: ", i, " error: ", e)
			error_list.append(e)
		print ("After ", NUM_TRIALS, " for group size:", j, " Error: ", np.mean(error_list))

def main():
	multirun()

if __name__=='__main__':
	main()
