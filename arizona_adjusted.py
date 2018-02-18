import pandas as pd
import math
from localize.localize import *

import sys
from argparse import ArgumentParser

#FILE = "data11"


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


def deg2rad(deg):
	return deg * (math.pi/180)

def  getDistanceFromLatLonInm(lat1,lon1,lat2,lon2):
	R = 6371*1000; # Radius of the earth in m
	dLat = deg2rad(lat2-lat1)
	dLon = deg2rad(lon2-lon1) 
	a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) 
	d = R * c # Distance in m
	return d

def calculate_x(row, origin_x, origin_y):
	dist = getDistanceFromLatLonInm(row['LAT'], origin_y, origin_x, origin_y)
	return dist

def calculate_y(row, origin_x, origin_y):
	dist = getDistanceFromLatLonInm(origin_x, row['LONG'], origin_x, origin_y)
	return dist


def read_transmitters(file, id):
	f = open(file, 'r')
	lines = f.readlines()
	for line in lines[1:]:
		fileid, lat, long = line.split(",")
		if fileid == id:
			return float(lat), float(long)
	return -1, -1

def experiment(FILE, radius, INTERATIONS, NUM_FALSE, gsize, adjust):
	#df = pd.read_csv("CentralParking-Downtown-Data-TerrainType-USGS - NED 1 - 30m_header_Clutter_Categories.csv")
	df = pd.read_csv(FILE + ".csv")

	df = df[['LAT','LONG', "Meas PWR (dBm)"]]
	transmitter_x, transmitter_y = read_transmitters("transmitters.csv", FILE)
	df = df[df.apply(lambda x: getDistanceFromLatLonInm(x['LAT'], x['LONG'], transmitter_x, transmitter_y) > 100.0, axis=1)]
	sampled = df.sample(len(df))
	#print (sampled)


	# pick the maxima and receivers around it
	max_index = sampled["Meas PWR (dBm)"].idxmax(axis=1)
	mlat = df['LAT'][max_index]
	mlon = df['LONG'][max_index]
	print ("Maxima at: ", mlat, mlon)

	sampled = sampled[sampled.apply(lambda x: getDistanceFromLatLonInm(x['LAT'], x['LONG'], mlat, mlon) < radius, axis=1)]
	print ("Samples in sampled: ", len(sampled))

	min_lat, max_lat = min(sampled['LAT']), max(sampled['LAT']) # this will become the origin
	print(min_lat, max_lat)

	min_long, max_long = min(sampled['LONG']), max(sampled['LONG'])
	print(min_long, max_long)

	sampled['X'] = sampled.apply(lambda row: calculate_x(row, min_lat, min_long),axis=1)
	sampled['Y'] = sampled.apply(lambda row: calculate_y(row, min_lat, min_long),axis=1)
	#print(sampled)
	print("area: ", max(sampled['X']) - min(sampled['X']), max(sampled['Y']) - min(sampled['Y']))
	grid_centers = calculate_grid_centers(max(sampled['X']), max(sampled['Y']), min(sampled['X']), min(sampled['Y']), gsize)
	# prepare receiver list
	receiver_list = []

	for index, row in sampled.iterrows():
		receiver_list.append([row['X'], row['Y'], row['Meas PWR (dBm)']])
	x, y = localize(receiver_list, grid_centers)[0]


	#INTERATIONS = 100
	### privacy enabled localization
	error_list_adjusted = []
	transmitter_x, transmitter_y = read_transmitters("transmitters.csv", FILE)
	x_trans = getDistanceFromLatLonInm(transmitter_x, min_long, min_lat, min_long)
	y_trans = getDistanceFromLatLonInm(min_lat, transmitter_y, min_lat, min_long)

	
	print ("error: ", edist(x_trans, y_trans, x, y))

	for m in range(INTERATIONS):
		print ("Interation number: ", m+1)
		if adjust:
			x_coor = []
			y_coor = []

			#NUM_FALSE = 400
			for i in range(NUM_FALSE):
				x_coor.append(random.uniform(min(sampled['X']), max(sampled['X'])))
				y_coor.append(random.uniform(min(sampled['Y']), max(sampled['Y'])))

			rss_values = estimate_rss(receiver_list, x_coor, y_coor)
		else:
			x_coor = [r[0] + random.uniform(-500, 500) for r in receiver_list]
			y_coor = [r[1] + random.uniform(-500, 500) for r in receiver_list]
			rss_values = [x[-1] for x in receiver_list]

		adjusted_recv_list = []
		for m in range(len(rss_values)):
			adjusted_recv_list.append([x_coor[m], y_coor[m], rss_values[m]])

		x_adj, y_adj = localize(adjusted_recv_list, grid_centers)[0]

		
		error_list_adjusted.append(edist(x_trans, y_trans, x_adj, y_adj))

	if error_list_adjusted:
		print("Average Adjusted error: ", np.mean(error_list_adjusted))

def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--datasource", dest="datasource",\
            required=True, type=str,  help="the data file to read from")
    parser.add_argument("-n", "--numiters", dest="numiters", default=50,\
            type=int, help="number of iterations to run")
    parser.add_argument("-f", "--fnum", dest="fnum", default=400,\
            type=int, help="number of false locations to sample")
    parser.add_argument("-r", "--radius", dest="radius", default=300, type=int,\
            help="radius around maxima for receiver selection")
    
    parser.add_argument("-g", "--gsize", dest="gsize", default=50, type=int,\
            help="the grid size for SPLOT")
    parser.add_argument("-a", "--adjust", dest="adjust", action="store_true",\
            help="adjust the rss value or not")
    
    args, other_args = parser.parse_known_args()
    print("running for ", args.datasource, " radius: ", args.radius, " for ", args.numiters, " iterations with", args.fnum, " false locations and grid size: ", args.gsize)
    experiment(args.datasource, args.radius, args.numiters, args.fnum, args.gsize, args.adjust)

if __name__=='__main__':
	main()
