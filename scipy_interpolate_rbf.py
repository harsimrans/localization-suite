from scipy.interpolate import Rbf
import pandas as pd
import numpy as np
import math


def get_estimated_rss(xr, yr, rssr, xe, ye, sv=0):
	''' takes as input reference points in the form
		xr: x-coordinate of location
		yr: y-coordinate of location
		rssr: rss at the x, y coordinate
		xe: x-coordinate list to estimated
		ye: y-coordinate list of points to be estimated
		smooth: level of smoothening required
	'''
	model = Rbf(xr, yr, rssr, smooth=sv)
	out = model(xe, ye)
	return out

def MSE(x, y):
   error = 0.0
   for i in range(len(x)):
	 error += (x[i] - y[i])**2
   return error

def MAE(x, y):
	error = 0.0
	for i in range(len(x)):
		error += math.fabs(x[i] - y[i])
	return error / len(x)

def experiment(loc, rss, txid):
	print "For transmitter: ", txid
	# pick set of anchor nodes randomly
	for i in range(27, len(loc)-2):
		
		idx = np.random.choice(np.arange(len(loc)), i+1, replace=False)
		
		loc_anchor_x = []
		loc_anchor_y = []
		rss_anchor = []
		
		loc_pred_x = []
		loc_pred_y = []
		rss_pred = []

		for j in range(len(loc)):
			if j == txid:
				continue
			if j in idx:
				loc_anchor_x.append(loc[0][j])
				loc_anchor_y.append(loc[1][j])
				rss_anchor.append(rss[txid][j])
			else:
				loc_pred_x.append(loc[0][j])
				loc_pred_y.append(loc[1][j])
				rss_pred.append(rss[txid][j])

		smooth_vals = [1,3,5,7,10]
		for smooth_val in smooth_vals:
			MAEs = []
			MSEs = []
			#print len(loc_anchor_x) ,len(loc_anchor_y), len(rss_anchor), len(loc_pred_x), len(loc_pred_y)
			output = get_estimated_rss(loc_anchor_x, loc_anchor_y, rss_anchor, loc_pred_x, loc_pred_y, smooth_val)
			MAEs.append(MAE(output, rss_pred))
			MSEs.append(MSE(output, rss_pred))
		print "\t", "MSE: ", min(MSEs), "MAE: ", min(MAEs), " Num anchor nodes: ", len(loc_anchor_x)


def main():
	# read the files
	loc = pd.read_csv("loc_val.csv", header=None)
	rss = pd.read_csv("rss_val.csv", header=None)
	for i in range(44):
		experiment(loc, rss, i)

if __name__=='__main__':
	main()