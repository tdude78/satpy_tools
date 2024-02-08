import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G, M_earth, R_earth
from scipy.special import jv
from sgp4 import exporter
from sgp4.api import SGP4_ERRORS, WGS72, Satrec

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from satpy_tools.constants import JULIAN_FIX, MU, NOW_MJD, RE, SGP4_JDOFFSET
from satpy_tools.conversions import cart2kep


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

        if isinstance(elements, np.ndarray):
            # convert elements [a, e, i, raan, argp, M] to [ecco, argpo, inclo, mo, no_kozai, nodeo]
            ecco  = elements[1]
            argpo = elements[4]
            inclo = elements[2]
            mo    = elements[5]
            no_kozai = np.sqrt(MU/elements[0]**3)*60
            nodeo = elements[3]
            
            try:
                B_star = elements[6]
            except IndexError:
                B_star = 0

            elems    = np.array([ecco, argpo, inclo, mo, no_kozai, nodeo])
            elements = elems
            if deg:
                elements[1] = np.deg2rad(elements[1])
                elements[2] = np.deg2rad(elements[2])
                elements[3] = np.deg2rad(elements[3])
                # elements[4] = np.deg2rad(elements[4])
                elements[5] = np.deg2rad(elements[5])

            if MJD is None:
                MJD = NOW_MJD

            self.jd = MJD + JULIAN_FIX
            jd_sgp4 = MJD + JULIAN_FIX - SGP4_JDOFFSET

            self.satellite = Satrec()
            self.satellite.sgp4init(
                WGS72, 'i', 1, jd_sgp4, B_star, 0, 0, *elements
            )
        # check if is tuple of strings
        elif isinstance(elements, tuple):
            self.satellite = Satrec.twoline2rv(*elements)
            self.jd  = self.satellite.jdsatepoch + self.satellite.jdsatepochF
        else:
            raise ValueError("Invalid input for elements. Must be a numpy array of orbital elements or a tuple of strings that are the TLE.")



    def propagate_to(self, time_days:float):
        time_days += self.satellite.jdsatepoch + self.satellite.jdsatepochF
        e, r, v    = self.satellite.sgp4(time_days, 0)
        if e != 0:
            raise RuntimeError(SGP4_ERRORS[e])
        state = np.concatenate((r, v))
        return state
    

    def propagate_step(self, time_days:float, timestep_s:float):
        timestep_days = timestep_s/86400
        ts            = np.arange(0, time_days+timestep_days, timestep_days)
        states        = np.zeros((len(ts), 7))
        mjd           = self.jd - JULIAN_FIX
        for i, t in enumerate(ts):
            try:
                state = self.propagate_to(t)
            except RuntimeError as e:
                print(e)
                states = states[:i,:]
                break
            t = np.array([mjd + t])
            states[i,:] = np.concatenate((t, state))
        return states
    

    def propagate_step_update(self, time_days:float, timestep_s:float):
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

    elements  = [6738.0,  0.0001217,  51.6398, 179.7719, 23.6641,  73.5536]
    days_prop = 180/60/24

    sat = SGP4SAT(elements)

    start = timeit.default_timer()
    states = sat.propagate_step(days_prop, 1)
    end = timeit.default_timer()
    print("Time: ", end-start, " seconds")

    rs = states[:,:3]
    rs = np.linalg.norm(rs, axis=1)

    rs = rs - RE
    rs = rs[::100]

    plt.figure()
    plt.plot(rs)
    plt.show()


    # print("Time: ", end-start, " seconds")

