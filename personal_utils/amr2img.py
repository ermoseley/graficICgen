#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import miniramses as ram
from matplotlib import colormaps
from colormaps import cmaps
valid_cmaps = ['cmapkk1', 'cmapkk2', 'cmapkk3', 'cmapkk4', 
                'cmapkk5', 'cmapkk6', 'cmapkk7', 'cmapkk8', 'cmapkk9', 'cmapkk10']
for cmap in valid_cmaps:
    colormaps.register(cmaps(cmap),force=True)

# Check if we should use non-interactive backend
import sys
if "--no-display" in sys.argv:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    print("Using non-interactive backend for batch processing")

parser = argparse.ArgumentParser()
parser.add_argument("nout", help="enter output number")
parser.add_argument("--path", help="specify a path")
parser.add_argument("--log", help="plot log variable",action="store_true")
parser.add_argument("--out", help="output a png image")
parser.add_argument("--prefix", help="specify a file prefix")
parser.add_argument("--col", help="choose the color map")
parser.add_argument("--min", help="specify a minimum variable value for colorbar")
parser.add_argument("--max", help="specify a maximum variable value for colorbar")
parser.add_argument("--var", help="specify a variable number")
parser.add_argument("--xcen", help="specify the image center x-coordinate")
parser.add_argument("--ycen", help="specify the image center y-coordinate")
parser.add_argument("--zcen", help="specify the image center z-coordinate")
parser.add_argument("--rad", help="specify the image radius")
parser.add_argument("--clump", help="specify if clumps are overplotted")
parser.add_argument("--sink", help="specify if sinks are overplotted")
parser.add_argument("--dir", help="specify the projection axis")
parser.add_argument("--grid", help="overlay the AMR grid",action="store_true")
parser.add_argument("--no-display", help="prevent GUI display (useful for batch processing)",action="store_true")
args = parser.parse_args()
# path the the file
path = args.path
prefix = args.prefix
ivar = args.var
vmin = args.min
vmax = args.max
col = args.col
radius = args.rad
xcenter = args.xcen
ycenter = args.ycen
zcenter = args.zcen
clump = args.clump
sink = args.sink
axis = args.dir
log = args.log
grid = args.grid
no_display = args.no_display

# Convert vmin and vmax to float if provided
if vmin is not None:
    vmin = float(vmin)
if vmax is not None:
    vmax = float(vmax)

grid0 = None
if grid:
    grid0=1
if clump==None:
    clump=False
if sink==None:
    sink=False
if axis==None:
    axis="z"
if xcenter==None:
    xcenter=None
else:
    xcenter=float(xcenter)
if ycenter==None:
    ycenter=None
else:
    ycenter=float(ycenter)
if zcenter==None:
    zcenter=None
else:
    zcenter=float(zcenter)
if radius==None:
    radius=None
else:
    radius=float(radius)
center=np.array([xcenter,ycenter,zcenter])

log0=None
if log:
    log0=1

if ivar==None:
    ivar=0
else:
    ivar=int(ivar)-1
if prefix==None:
    prefix="hydro"
if path==None:
    path="./"
else:
    path=path+"/"
if prefix=="hydro":
    isort=0
if prefix=="peak":
    isort=1
if prefix=="grav":
    isort=0
if prefix=="rt":
    isort=0

nout = args.nout
print("Reading output number ",nout)

if axis=="x":
    ii=2; jj=3
if axis=="y":
    ii=1; jj=3
if axis=="z":
    ii=1; jj=2

c=ram.rd_cell(nout,path=path,prefix=prefix,center=center,radius=radius)
kwargs={}
if col is not None:
    kwargs["cmap"]=col
# Create visualization with specified colorbar limits (vmin, vmax)
ram.visu(c.x[ii-1],c.x[jj-1],c.dx,c.u[ivar],sort=c.u[isort],log=log0,vmin=vmin,vmax=vmax,grid=grid0,**kwargs)

if clump:
    h=ram.rd_clump(nout)
    if radius is not None:
        r = np.sqrt((h.x-center[0])**2+(h.y-center[1])**2+(h.z-center[2])**2)
        nn = np.count_nonzero(r < radius)
        xx = h.x[r < radius]
        yy = h.y[r < radius]
        zz = h.z[r < radius]
        mm = h.mass[r < radius]
    else:
        xx = h.x
        yy = h.y
        zz = h.z
    if axis=="x":
        plt.plot(yy,zz,'r.')
    if axis=="y":
        plt.plot(xx,zz,'r.')
    if axis=="z":
        plt.plot(xx,yy,'r.')

if sink:
    s=ram.rd_part(nout,sink=True)
    if radius is not None:
        r = np.sqrt((s.xp[0]-center[0])**2+(s.xp[1]-center[1])**2+(s.xp[2]-center[2])**2)
        nn = np.count_nonzero(r < radius)
        xx = s.xp[0][r < radius]
        yy = s.xp[1][r < radius]
        zz = s.xp[2][r < radius]
        mm = s.mp[r < radius]
    else:
        xx = s.xp[0]
        yy = s.xp[1]
        zz = s.xp[2]
    if axis=="x":
        plt.plot(yy,zz,'r.')
    if axis=="y":
        plt.plot(xx,zz,'r.')
    if axis=="z":
        plt.plot(xx,yy,'r.')

if args.out:
    plt.savefig(args.out)

if not no_display:
    plt.show()
else:
    plt.close

