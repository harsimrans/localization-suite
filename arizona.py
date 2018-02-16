import pandas as pd
import math
from localize.localize import *


FILE = "data1"

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

sampled = sampled[sampled.apply(lambda x: getDistanceFromLatLonInm(x['LAT'], x['LONG'], mlat, mlon) < 1000.0, axis=1)]
print ("Samples in sampled: ", len(sampled))

min_lat, max_lat = min(sampled['LAT']), max(sampled['LAT']) # this will become the origin
print(min_lat, max_lat)

min_long, max_long = min(sampled['LONG']), max(sampled['LONG'])
print(min_long, max_long)

sampled['X'] = sampled.apply(lambda row: calculate_x(row, min_lat, min_long),axis=1)
sampled['Y'] = sampled.apply(lambda row: calculate_y(row, min_lat, min_long),axis=1)
print(sampled)

grid_centers = calculate_grid_centers(max(sampled['X']), max(sampled['Y']), min(sampled['X']), min(sampled['Y']), 10.0)
# prepare receiver list
receiver_list = []

for index, row in sampled.iterrows():
	receiver_list.append([row['X'], row['Y'], row['Meas PWR (dBm)']])
x, y = localize(receiver_list, grid_centers)[0]

transmitter_x, transmitter_y = read_transmitters("transmitters.csv", FILE)

x_trans = getDistanceFromLatLonInm(transmitter_x, min_long, min_lat, min_long)
y_trans = getDistanceFromLatLonInm(min_lat, transmitter_y, min_lat, min_long)

print("area: ", max(sampled['X']) - min(sampled['X']), max(sampled['Y']) - min(sampled['Y']))
print ("error: ", edist(x_trans, y_trans, x, y))
print(edist(0, 0, 1,1))
