'''Script for analyzing MCell molecule location data for calcium diffusion from
 a plane half-space
 - Current data is run without temperature adjustments (may have different
 amount of calcium that enters; diffusion is unaffected
 '''

# import packages
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from scipy import stats


def time_hist(iter_num, plot=True):
    mcell_viz_dir = "/Users/margotwagner/projects/mcell/simple_geom/" \
                    "infinite_space/half_space_plane_files/mcell/output_data/viz_data_ascii"

    ca_hist = []
    for seed in sorted(os.listdir(mcell_viz_dir)):
        iter_file = os.path.join(mcell_viz_dir, seed, 'Scene.ascii.{}.dat'.format(iter_num))
        loc_data = pd.read_csv(iter_file, delim_whitespace=True, header=None,
                           names=['type', 'id', 'x', 'y', 'z', 'norm_x', 'norm_y', 'norm_z'])
        # select only calcium
        loc_data = loc_data[loc_data['type'] == 'ca']

        # radius from x, y, and z coordinates
        loc_data['r'] = np.sqrt(loc_data['x'] ** 2 + loc_data['y'] ** 2 + loc_data['z'] ** 2)

        for value in loc_data['r'].values:
            ca_hist.append(value)

    if plot:
        #plt.hist(ca_hist, bins=250, color='C0')
        sns.distplot(ca_hist, bins=100)
        plt.show()

    return ca_hist

# plot for multiple time points
nums = ['00500','01910','04280', '10000']
data = []
for num in nums:
    ca_hist = time_hist(num, plot=False)
    sns.distplot(ca_hist, label=num)

plt.xlabel('Radius')
plt.ylabel('Frequency')
plt.title('Calcium frequency with radius')
plt.legend()
plt.show()


