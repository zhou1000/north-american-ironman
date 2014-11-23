#! /usr/bin/python

# count the number of rows of a file
# Yiqian
# usage: python subsample.py input output ratio
# see also shell command: wc -l file

import sys

if len(sys.argv)!=2:
    raise Exception("\n USAGE: python countRow.py file\n")

file = sys.argv[1]
with open(file,'r') as f:
    for i, line in enumerate(f):
        pass
    print "{0} has {1} lines".format(file, i+1)
