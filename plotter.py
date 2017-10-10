#!/usr/bin/env python

import matplotlib.pyplot as plt
import math
import pandas as pd

# if 25 set this to 24 because it's zero based indexing


NUMBER = 43

df = pd.read_csv("loc_val.csv", header=None)
df2 = pd.read_csv("rss_val.csv", header=None)

fig, ax = plt.subplots()
ax.scatter(df[0], df[1], color="blue")

ax.scatter(df[0], df[1])

def cdist(x1, y1, x2, y2):
  return math.sqrt((x1-x2)**2 + (y1-y2)**2)
 

for i, txt in enumerate(df2[NUMBER]):
    label = str(i+1) + ' ' + str(round(cdist(df[0][i], df[1][i], df[0][NUMBER], df[1][NUMBER]), 2)) + " " + str(round(float(txt), 2))
    ax.annotate(label, (df[0][i], df[1][i]))

plt.xlabel("X-coordinate (m)")
plt.ylabel("Y-coordinate (m)")
plt.title("RSS values with "+ str(NUMBER) + " as transmitter, (id, distance, rss)")
plt.show()