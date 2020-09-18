import pandas as pd
import re
from statistics import mean,stdev

dir = '/Users/margotwagner/projects'

with open('/Users/margotwagner/projects/Benchmarks') as file:
    data = file.readlines()

data = [x.strip() for x in data]

vdcc = []
calb = []
active_list = vdcc
for line in data:
    if line == 'Calbindin':
        active_list = calb

    line = line.split(' ')

    if len(line) == 10:
        active_list.append(float(line[5]) + float(line[8]))


print("VDCC:", round(mean(vdcc),2), "s", u"\u00B1", round(stdev(vdcc),2),"s for", len(vdcc), "runs")
print("Calbindin:", round(mean(calb),2), "s", u"\u00B1", round(stdev(calb),2),"s for", len(calb), "runs")


