# /usr/bin/python
import csv
import random
import sys

# usage: python subsample.py input output ratio

inputfile = sys.argv[1]
outputfile = sys.argv[2]
ratio = float(sys.argv[3])

with open(outputfile,'w') as fout:
    writer = csv.writer(fout)
    with open(inputfile, 'r') as fin:
        for row in csv.reader(fin):
            if(random.random() < ratio): # select ratio randomly
                writer.writerow(row)

