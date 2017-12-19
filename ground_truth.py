#!/usr/bin/env python

#
# This script serves as base/ground truth. It does not
# add noisy to any of the transmitter locations.
#
# It uses localize module from the following paper titled:
#
# "Simultaneous Power-Based Localization of Transmitters 
# for Crowdsourced Spectrum Monitoring"
# by Khaledi et. al
#

from __future__ import print_function
from localize.localize import *
import os


def main():
	NUM_REC = 43
	# read the data
	loc, rss = read_dataset()
	recv_list, trans_list = get_receiver_snapshots(loc, rss, NUM_REC)
	grid_centers = calculate_grid_centers(10, 13, -5, -5, 1.0)

	error = []
	
	for i in range(len(recv_list)):
		tx_loc = localize(recv_list[i], grid_centers)[0]
		e = edist(tx_loc[0], tx_loc[1], trans_list[i][0], trans_list[i][1])
		error.append(e)

		print("Error:", "{0:.2f}".format(e), "m")
	print("Average Error: ", "{0:.2f}".format(np.mean(error)), "m")


if __name__=='__main__':
	main()
