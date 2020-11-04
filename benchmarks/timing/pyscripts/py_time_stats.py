#!/Library/Frameworks/Python.framework/Versions/3.8/bin/python3

import pandas as pd
import numpy as np
import os
import sys
import glob
from statistics import mean, stdev

os.chdir('/Users/margotwagner/projects/DetailedMarkov/benchmarks/timing/pyscripts')
sys.stdout = open("stats_pytime.txt", "w")

def stats_pytime(fname):
    runtimes = []
    for run in glob.glob(os.path.join(fname, '*.out')):
        time = np.loadtxt(run)
        runtimes.extend(time[:,1].tolist())

    avg = mean(runtimes)
    std = stdev(runtimes)

    print("{:.2f}".format(avg), u"\u00B1", "{:.2}".format(std), 's for', len(runtimes), 'trials of', fname.strip('./'))
    return mean, std


files = glob.glob('calb*')
files.extend(glob.glob('vdcc*'))
files.remove('calb_markov')

for file in sorted(files):
    stats_pytime(file)

sys.stdout.close()
