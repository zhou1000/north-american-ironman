#! /usr/bin/python

# subsample data randomly
# Yiqian
# usage: python subsample.py input output ratio

import random
import sys

if len(sys.argv)!=4:
    raise Exception("\n USAGE: python subsample.py input output ratio\n")

inputfile = sys.argv[1]
outputfile = sys.argv[2]
ratio = float(sys.argv[3])

with open(outputfile,'w') as fout:
    with open(inputfile, 'r') as fin:
        row = fin.next()
        fout.write(row) # keep the header
        for row in fin:
            if(random.random() < ratio): # select ratio randomly
                fout.write(row)

