#!/Library/Frameworks/Python.framework/Versions/3.8/bin/python3

import pandas as pd
import numpy as np
import os
import sys
import glob

os.chdir('/Users/margotwagner/projects/DetailedMarkov/benchmarks/flops/')
sys.stdout = open("flops_stats.txt", "w")

def stats_flops(fname):
    flops = pd.read_csv(fname, header=None).to_numpy(dtype=str)
    flops = np.char.strip(flops, chars='[]').astype('float')
    mean = np.mean(flops)
    std = np.std(flops)

    print("{:.3e}".format(mean), u"\u00B1", "{:.3e}".format(std), 'for', len(flops), 'trials of', fname.strip('./'))
    return mean, std

for file in sorted(glob.glob("bnch*.txt")):
    stats_flops(file)

sys.stdout.close()

