from datetime import datetime
import os

import juliandate
import numpy as np
import ray
from ray.exceptions import RaySystemError

ray.init()

def get_mem_func():
	MEMORY = ray.available_resources()['memory']
	try:
		CPUS = ray.available_resources()['CPU']
		MEM_PER_WORKER = (MEMORY / CPUS) * 0.8
	except KeyError:
		# get number of cores
		import multiprocessing
		CPUS = multiprocessing.cpu_count()
		MEM_PER_WORKER = (MEMORY / CPUS) * 0.8
	MEMORY = int(MEMORY)
	CPUS   = int(CPUS)
	MEM_PER_WORKER = int(MEM_PER_WORKER)
	return MEMORY, CPUS, MEM_PER_WORKER

try:
	MEMORY, CPUS, MEM_PER_WORKER = get_mem_func()
except RaySystemError:
	MEMORY, CPUS, MEM_PER_WORKER = get_mem_func()

# Earth shape parameters
MU    = (3.986004418e14)/(1e3)**3
RE    = 6371.000
RE_po = 6356.752
RE_eq = 6378.137
g  = lambda h: MU/(RE + h)**2

DU = 6378.145
TU = 806.8118744
DU_TU = DU/TU


MIN_V = np.sqrt(MU/(RE+250))
MAX_V = np.sqrt(MU/(35786))

SGP4_JDOFFSET = 2433281.5
JULIAN_FIX = 2400000.5
right_now  = datetime.now()
now_year, now_month, now_day = right_now.year, right_now.month, right_now.day
now_hr, now_min, now_sec  = right_now.hour, right_now.minute, right_now.second
NOW_MJD    = juliandate.from_gregorian(now_year,now_month,now_day,now_hr,now_min,now_sec) - JULIAN_FIX

MIN_PER_DAY   = 1440
SEC_PER_DAY   = 60 * MIN_PER_DAY

# Earth atmospheric & Chemistry Parameters
n0 = 1.00027717
R  = 287 # J/kg/K
p0 = 101325 # Pa
