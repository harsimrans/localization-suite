#!/usr/bin/env python

#
# It simply adds noise to the receiver locations
# add noisy to any of the transmitter locations.
#

from __future__ import print_function
from localize.localize import *
import os


def main():
	NUM_REC = 43
	NOISE_VAL = 10.0
	# read the data
	loc, rss = read_dataset()
	recv_list, trans_list = get_receiver_snapshots(loc, rss, NUM_REC)
	grid_centers = calculate_grid_centers(10, 13, -5, -5, 1.0)

	error = []
	
	for i in range(len(recv_list)):
		noisy_recv_list = add_location_noise(recv_list[i], NOISE_VAL)
		noisy_recv_list = add_location_noise_vary_privacy(recv_list[i], NOISE_VAL)
		noisy_recv_list = tweak_rss_powers(recv_list[i], NOISE_VAL)
		tx_loc = localize(noisy_recv_list, grid_centers)[0]
		e = edist(tx_loc[0], tx_loc[1], trans_list[i][0], trans_list[i][1])
		error.append(e)

		print("Error:", "{0:.2f}".format(e), "m")
	print("Average Error: ", "{0:.2f}".format(np.mean(error)), "m")


if __name__=='__main__':
	main()
