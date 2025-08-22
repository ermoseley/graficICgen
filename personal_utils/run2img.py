#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
from astropy.io import ascii
import argparse
import os
from matplotlib import colormaps
from colormaps import cmaps
valid_cmaps = ['cmapkk1', 'cmapkk2', 'cmapkk3', 'cmapkk4', 
                'cmapkk5', 'cmapkk6', 'cmapkk7', 'cmapkk8', 'cmapkk9', 'cmapkk10']
for cmap in valid_cmaps:
    colormaps.register(cmaps(cmap),force=True)

parser = argparse.ArgumentParser()
parser.add_argument("file", help="enter filename run.log")
parser.add_argument("--log", help="plot log variable",action="store_true")
parser.add_argument("--sym", help="use a circle for each cell",action="store_true")
parser.add_argument("--out", help="output a png image")
parser.add_argument("--no_display", help="do not display the image",action="store_true")
args = parser.parse_args()
print("Reading "+args.file)

# path the the file
path_to_output = args.file

# read log file
cmd="grep -n Output "+path_to_output+" > /tmp/out.txt"
os.system(cmd)
lines = ascii.read("/tmp/out.txt")
i = int(lines["col1"][-1][:-1])
n = int(lines["col3"][-1])

data = ascii.read(path_to_output,header_start=i-3,data_start=i-2,data_end=i+n-2)

if args.sym:
    plt.plot(data["x"],data["d"],"o")
else:
    plt.plot(data["x"],data["d"])
if args.log:
    plt.yscale("log")

if args.out:
    plt.savefig(args.out)
if not args.no_display:
    plt.show()
else:
    plt.close()


