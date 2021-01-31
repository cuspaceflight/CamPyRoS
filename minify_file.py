import PseudoNetCDF as pnc
import os
import numpy as np
import ray
files=os.listdir("data/wind/gfs/0p25/")

@ray.remote
def process(file):
    if file[-4:]==".arl":
        forcast=pnc.pncopen("data/wind/gfs/0p25/%s"%file, format="arlpackedbit")
        forcast=forcast.interpDimension("x",np.array([-4.01])).interpDimension("y",np.array([52.42]))
        forcast.save("data/wind/gfs/0p25/capel-dewi/%s.nc"%file[:-4], format="NETCDF4_CLASSIC")

ray.init(num_cpus=2)#This is limited by the memory consumption which peaks above 16GB with 2 cores
for file in files:
    process.remote(file)

input()