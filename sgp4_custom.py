import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G, M_earth, R_earth
from scipy.special import jv
from sgp4 import exporter
from sgp4.api import SGP4_ERRORS, WGS72, Satrec

from constants import MU, RE, NOW_MJD, SGP4_JDOFFSET, JULIAN_FIX
from conversions import cart2kep, kep2cart

# 02/07/24

class SGP4SAT:
    def __init__(self, elements:np.ndarray, MJD=NOW_MJD, deg=True):
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

        self.satellite = Satrec()

        jd_sgp4 = MJD + JULIAN_FIX - SGP4_JDOFFSET
        self.satellite.sgp4init(
            WGS72, 'i', 1, jd_sgp4, B_star, 0, 0, *elements
        )


    def _propagate(self, time:float):
        e, r, v = self.satellite.sgp4(time, 0)
        if e != 0:
            raise RuntimeError(SGP4_ERRORS[e])
        state = np.concatenate((r, v))
        return state
    

    def propagate(self, time_days:float, timestep_s:float):
        timestep_days = timestep_s/86400
        ts            = np.arange(0, time_days, timestep_days)
        states        = np.zeros((len(ts), 6))

        for i, t in enumerate(ts):
            state = self._propagate(t)
            states[i,:] = state

        elements = cart2kep(states[-1,:])
        self.satellite = SGP4SAT(elements).satellite

        return states


if __name__ == "__main__":
    # make this take command line arguments
    import sys
    import timeit

    elements  = [6738.0,  0.0001217,  51.6398, 179.7719, 23.6641,  73.5536]
    days_prop = 180/60/24

    sat = SGP4SAT(elements)

    start = timeit.default_timer()
    states = sat.propagate(days_prop, 1)
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

