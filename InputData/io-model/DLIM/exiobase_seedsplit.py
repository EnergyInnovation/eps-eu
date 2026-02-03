# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:59:50 2025

@author: DanO'Brien
"""

import pymrio
import numpy as np

# EU-27 definition
EU = {
    "AT","BE","BG","CY","CZ","DE","DK","EE","ES","FI","FR","GR","HR","HU",
    "IE","IT","LT","LU","LV","MT","NL","PL","PT","RO","SE","SI","SK"
}


# parse the EXIOBASE system
exio3 = pymrio.parse_exiobase3(path="C:/Users/DanO'Brien/Downloads/IOT_2022_pxp.zip")

# Get regions and sectors
exio3.get_sectors()
exio3.get_regions()

# Read regions directly from pymrio object
regions = list(exio3.regions)

# Aggregated region labels
agg_regions = ["EU", "ROW"]

# Build aggregation matrix
reg_agg_matrix = np.zeros((len(agg_regions), len(regions)), dtype=int)

for j, r in enumerate(regions):
    if r in EU:
        reg_agg_matrix[0, j] = 1   # EU
    else:
        reg_agg_matrix[1, j] = 1   # ROW

print(reg_agg_matrix)

exio3.aggregate(region_agg=reg_agg_matrix)


exio3.calc_all()

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))
plt.imshow(exio3.A, vmax=1e-2)
plt.xlabel("Countries - sectors")
plt.ylabel("Countries - sectors")
plt.show()
"""
with open("regions.txt", "w") as f:
    f.write("\n".join(exio3.regions))

with open("sectors.txt", "w") as f:
    f.write("\n".join(exio3.sectors))

"""