import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv
from sgp4 import exporter
from sgp4.api import SGP4_ERRORS, WGS72, Satrec, jday

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from satpy_tools.constants import JULIAN_FIX, MU, NOW_MJD, RE, SGP4_JDOFFSET
    from satpy_tools.conversions import cart2kep
except ImportError:
    from constants import JULIAN_FIX, MU, NOW_MJD, RE, SGP4_JDOFFSET
    from conversions import cart2kep


class SGP4SAT:
    def __init__(self, elements, MJD=None, deg=True):
        '''
        >>> satellite2 = Satrec()
        >>> satellite2.sgp4init(
        ...     WGS72,                # gravity model
        ...     'i',                  # 'a' = old AFSPC mode, 'i' = improved mode
        ...     25544,                # satnum: Satellite number
        ...     25545.69339541,       # epoch: days since 1949 December 31 00:00 UT
        ...     3.8792e-05,           # bstar: drag coefficient (1/earth radii)
        ...     0.0,                  # ndot: ballistic coefficient (radians/minute^2)
        ...     0.0,                  # nddot: mean motion 2nd derivative (radians/minute^3)
        ...     0.0007417,            # ecco: eccentricity
        ...     0.3083420829620822,   # argpo: argument of perigee (radians)
        ...     0.9013560935706996,   # inclo: inclination (radians)
        ...     1.4946964807494398,   # mo: mean anomaly (radians)
        ...     0.06763602333248933,  # no_kozai: mean motion (radians/minute)
        ...     3.686137125541276,    # nodeo: R.A. of ascending node (radians)
        ... )
        '''

        if isinstance(elements, np.ndarray) or isinstance(elements, list):
            # convert elements [a, e, i, raan, argp, M] to [ecco, argpo, inclo, mo, no_kozai, nodeo]
            e     = elements[1]
            argpo = elements[4]
            i     = elements[2]
            M     = elements[5]
            n_min = np.sqrt(MU/elements[0]**3)*60
            raan  = elements[3]
            
            try:
                B_star = elements[6]
            except IndexError:
                B_star = 0
            if deg:
                i         = np.deg2rad(i)
                argpo     = np.deg2rad(argpo)
                raan      = np.deg2rad(raan)
                M         = np.deg2rad(M)
            elems    = np.array([e, argpo, i, M, n_min, raan])

            if MJD is None:
                MJD = NOW_MJD

            self.jd = MJD + JULIAN_FIX
            jd_sgp4 = MJD + JULIAN_FIX - SGP4_JDOFFSET

            self.satellite = Satrec()
            self.satellite.sgp4init(
                WGS72, 'i', 1, jd_sgp4, B_star, 0, 0, *elems
            )
        # check if is tuple of strings
        elif isinstance(elements, tuple):
            self.satellite = Satrec.twoline2rv(*elements)
            self.jd        = self.satellite.jdsatepoch + self.satellite.jdsatepochF
        else:
            raise ValueError("Invalid input for elements. Must be a numpy array of orbital elements or a tuple of strings that are the TLE.")


    def propagate_to(self, time_days:float):
        time_days += self.satellite.jdsatepoch + self.satellite.jdsatepochF
        e, r, v = self.satellite.sgp4(time_days, 0)
        if e != 0:
            raise RuntimeError(SGP4_ERRORS[e])
        return np.concatenate((r, v), axis=0)
    

    def propagate_step(self, time_days:float, timestep_s:float=0.1):
        timestep_days = timestep_s/86400

        ts      = np.arange(0, time_days+timestep_days, timestep_days)
        ts     += self.satellite.jdsatepoch + self.satellite.jdsatepochF
        e, r, v = self.satellite.sgp4_array(ts, np.zeros(ts.shape))
        if np.any(e != 0):
            raise RuntimeError(SGP4_ERRORS[e[e != 0]])
        states = np.concatenate((r, v), axis=1)
        return states
    
    # final note
    # working to get rid of step, combine propagate_to and propagate_step maybe?

    def propagate_step_update_DEP(self, time_days:float, timestep_s:float):
        states = self.propagate_step(time_days, timestep_s)

        mjd_f          = self.jd + time_days - JULIAN_FIX
        elements       = cart2kep(states[-1,:], deg=False)
        sat            = SGP4SAT(elements, MJD=mjd_f, deg=False)
        self.satellite = sat.satellite
        self.jd        = sat.jd
        return states


if __name__ == "__main__":
    # make this take command line arguments
    import sys
    import timeit

    with open('satpy_tools/equisat.tle', 'r') as f:
        lines = f.readlines()

    line1 = lines[0]
    line2 = lines[1]

    elements = (line1, line2)

    sat = SGP4SAT(elements)

    time_days = 180/60/24
    timestep_s = 60
    timestep_days = timestep_s/86400

    start = timeit.default_timer()
    states = sat.propagate_step_update_DEP(time_days, timestep_s)

    # elements  = [6738.0,  0.0001217,  51.6398, 179.7719, 23.6641,  73.5536]

    # sat = SGP4SAT(elements)

    # time_days = 180/60/24
    # timestep_s = 60
    # timestep_days = timestep_s/86400

    # start = timeit.default_timer()
    # states = sat.propagate_step(time_days, timestep_s)
    # end = timeit.default_timer()
    # print("Time: ", end-start, " seconds")


    # # print("Time: ", end-start, " seconds")

