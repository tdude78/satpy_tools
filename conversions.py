import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import juliandate
import numpy as np
from scipy.special import jv

try:
    from satpy_tools.constants import JULIAN_FIX, MU
except ImportError:
    from constants import JULIAN_FIX, MU

from collections import namedtuple
from contextlib import contextmanager
from copy import deepcopy
from math import atan2, floor, fmod, isinf, isnan

import numpy as np
from numpy import arccos as acos
from numpy import cos, dot, sin, sqrt
from numpy.linalg import norm
from scipy.constants import pi


def cart2kep(state, deg=True):
    # https://github.com/RazerM/orbital/blob/0.7.0/orbital/utilities.py#L252
    mu    = MU

    r = state[0:3]
    v = state[3:6]

    h = np.cross(r, v)
    n = np.cross([0, 0, 1], h)

    ev = 1 / mu * ((norm(v) ** 2 - mu / norm(r)) * r - dot(r, v) * v)

    E = norm(v) ** 2 / 2 - mu / norm(r)

    a = -mu / (2 * E)
    e = norm(ev)

    SMALL_NUMBER = 1e-15

    # Inclination is the angle between the angular
    # momentum vector and its z component.
    i = acos(h[2] / norm(h))

    if abs(i - 0) < SMALL_NUMBER:
        # For non-inclined orbits, raan is undefined;
        # set to zero by convention
        raan = 0
        if abs(e - 0) < SMALL_NUMBER:
            # For circular orbits, place periapsis
            # at ascending node by convention
            arg_pe = 0
        else:
            # Argument of periapsis is the angle between
            # eccentricity vector and its x component.
            arg_pe = acos(ev[0] / norm(ev))
    else:
        # Right ascension of ascending node is the angle
        # between the node vector and its x component.
        raan = acos(n[0] / norm(n))
        if n[1] < 0:
            raan = 2 * pi - raan

        # Argument of periapsis is angle between
        # node and eccentricity vectors.
        arg_pe = acos(dot(n, ev) / (norm(n) * norm(ev)))

    if abs(e - 0) < SMALL_NUMBER:
        if abs(i - 0) < SMALL_NUMBER:
            # True anomaly is angle between position
            # vector and its x component.
            nu = acos(r[0] / norm(r))
            if v[0] > 0:
                nu = 2 * pi - nu
        else:
            # True anomaly is angle between node
            # vector and position vector.
            nu = acos(dot(n, r) / (norm(n) * norm(r)))
            if dot(n, v) > 0:
                nu = 2 * pi - nu
    else:
        if ev[2] < 0:
            arg_pe = 2 * pi - arg_pe

        # True anomaly is angle between eccentricity
        # vector and position vector.
        nu = acos(dot(ev, r) / (norm(ev) * norm(r)))

        if dot(r, v) < 0:
            nu = 2 * pi - nu
    
    if deg:
        i     = np.rad2deg(i)
        raan  = np.rad2deg(raan)
        arg_pe = np.rad2deg(arg_pe)
        nu     = np.rad2deg(nu)
            
        i     = np.mod(i, 360)
        raan  = np.mod(raan, 360)
        arg_pe = np.mod(arg_pe, 360)
        nu     = np.mod(nu, 360)
    else:
        i     = np.mod(i, 2*np.pi)
        raan  = np.mod(raan, 2*np.pi)
        arg_pe = np.mod(arg_pe, 2*np.pi)
        nu     = np.mod(nu, 2*np.pi)
    elems = np.array([a, e, i, raan, arg_pe, nu], dtype=np.float64)

    return elems


def kep2cart(state, deg=True):
    a, e, i, raan, arg_pe, M = state
    if deg:
        i     = np.deg2rad(i)
        raan  = np.deg2rad(raan)
        arg_pe = np.deg2rad(arg_pe)
        M     = np.deg2rad(M)

    mu    = MU
    omega = arg_pe
    Omega = raan

    E = M
    E_prev = 0
    for n in range(100):
        if n == 0:
            continue
        
        # E += jv(n, n*e)*np.sin(n*M)
        E -= (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
        if np.abs(E - E_prev) < 1e-12:
            break
        E_prev = E

    nu = 2 * atan2(sqrt(1 + e) * sin(E / 2), sqrt(1 - e) * cos(E / 2))

    rc = a*(1 - e*np.cos(E))

    ox = rc*np.cos(nu)
    oy = rc*np.sin(nu)

    ox_dot = (np.sqrt(mu*a)/rc)*(-np.sin(E))
    oy_dot = (np.sqrt(mu*a)/rc)*(np.sqrt(1-e**2)*np.cos(E))

    r1 = ox*(np.cos(omega)*np.cos(Omega) - np.sin(omega)*np.sin(Omega)*np.cos(i)) - oy*(np.sin(omega)*np.cos(Omega) + np.cos(omega)*np.sin(Omega)*np.cos(i))
    r2 = ox*(np.cos(omega)*np.sin(Omega) + np.sin(omega)*np.cos(Omega)*np.cos(i)) - oy*(np.sin(omega)*np.sin(Omega) - np.cos(omega)*np.cos(Omega)*np.cos(i))
    r3 = ox*(np.sin(omega)*np.sin(i)) + oy*(np.cos(omega)*np.sin(i))

    v1 = ox_dot*(np.cos(omega)*np.cos(Omega) - np.sin(omega)*np.sin(Omega)*np.cos(i)) - oy_dot*(np.sin(omega)*np.cos(Omega) + np.cos(omega)*np.sin(Omega)*np.cos(i))
    v2 = ox_dot*(np.cos(omega)*np.sin(Omega) + np.sin(omega)*np.cos(Omega)*np.cos(i)) - oy_dot*(np.sin(omega)*np.sin(Omega) - np.cos(omega)*np.cos(Omega)*np.cos(i))
    v3 = ox_dot*(np.sin(omega)*np.sin(i)) + oy_dot*(np.cos(omega)*np.sin(i))

    state = np.array([r1, r2, r3, v1, v2, v3], dtype=np.float64)
    return state


def parse_TLE(line1, line2, deg=True):
    line1 = line1.split()
    line2 = line2.split()

    # line 1
    sat_num = str(line1[1])
    # classification = str(line1[2])
    int_desig = line1[2]
    time = line1[3].split('.')
    epoch_year = int(time[0][0:2])
    if epoch_year > 56:
        epoch_year = epoch_year + 1900
    else:
        epoch_year = epoch_year + 2000

    epoch_day_int = float(time[0][2:])
    epoch_dayfrac = int(time[1])
    epoch_day = float(str(epoch_day_int) + str(epoch_dayfrac))
    MJD = juliandate.from_gregorian(epoch_year, 1, epoch_day) - JULIAN_FIX

    INC = float(line2[2])
    RAAN = float(line2[3])
    ECC = float("0." + line2[4])
    AOP = float(line2[5])
    MA = float(line2[6])
    MM = float(line2[7])

    SMA = (MU/(2*np.pi*MM/(24*3600))**2)**(1/3)

    if not deg:
        INC = np.deg2rad(INC)
        RAAN = np.deg2rad(RAAN)
        AOP = np.deg2rad(AOP)
        MA = np.deg2rad(MA)

    result = MJD, np.array([SMA, ECC, INC, RAAN, AOP, MA], dtype=np.float64)

    return result




def cart2kep_test(state, deg=True):
    # https://space.stackexchange.com/questions/1904/how-to-programmatically-calculate-orbital-elements-using-position-velocity-vecto
    eps  = 1e-10
    mu   = MU
    r    = state[0:3]
    v    = state[3:6]

    n = np.cross([0, 0, 1],r)

    h=np.cross(r,v)
    nhat=np.cross([0, 0, 1],h)

    evec = ((np.linalg.norm(v)^2-mu/np.linalg.norm(r))*r-np.dot(r,v)*v)/mu
    e = np.linalg.norm(evec)

    energy = np.linalg.norm(v)^2/2-mu/np.linalg.norm(r)

    if abs(e-1.0)>eps:
        a = -mu/(2*energy)
        p = a*(1-e^2)
    else:
        p = np.linalg.norm(h)^2/mu
        a = np.inf

    i = np.arccos(h[3-1]/np.linalg.norm(h))

    Omega = np.arccos(n[1-1]/np.linalg.norm(n))

    if n[2-1]<0:
        Omega = 360-Omega

    argp = np.arccos(np.dot(n,evec)/(np.linalg.norm(n)*e))

    if e[3-1]<0:
        argp = 360-argp

    nu = np.arccos(np.dot(evec,r)/(e*np.linalg.norm(r)))

    if np.dot(r,v)<0:
        nu = 360 - nu
def cart2kep_dep(state, deg=True):
    # https://web.archive.org/web/20160418175843/https://ccar.colorado.edu/asen5070/handouts/cart2kep2002.pdf
    # https://space.stackexchange.com/questions/19322/converting-orbital-elements-to-cartesian-state-vectors

    if len(state) == 7:
        state = state[1:]

    mu    = MU
    r_vec = state[0:3]
    v_vec = state[3:6]
    
    #1
    h_bar = np.cross(r_vec,v_vec)
    h     = np.linalg.norm(h_bar)
    #2
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    #3
    E = 0.5*(v**2) - mu/r
    #4
    a = -mu/(2*E)
    #5
    e = np.sqrt(1 - (h**2)/(a*mu))
    #6
    i = np.arccos(h_bar[2]/h)
    #7
    RAAN = np.arctan2(h_bar[0],-h_bar[1])
    #8
    #beware of division by zero here
    if np.sin(i) == 0:
        lat = 0
    else:
        lat = np.arctan2(np.divide(r_vec[2],(np.sin(i))), (r_vec[0]*np.cos(RAAN) + r_vec[1]*np.sin(RAAN)))
    #9
    p = a*(1-e**2)
    nu = np.arctan2(np.sqrt(p/mu) * np.dot(r_vec,v_vec), p-r)
    #10
    omega = lat - nu
    #11
    EA = 2*np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(nu/2))

    M = EA - e*np.sin(EA)
    if deg:
        i     = np.rad2deg(i)
        omega = np.rad2deg(omega)
        RAAN  = np.rad2deg(RAAN)
        M     = np.rad2deg(M)

        i     = np.mod(i, 360)
        omega = np.mod(omega, 360)
        RAAN  = np.mod(RAAN, 360)
        M     = np.mod(M, 360)
    else:
        i     = np.mod(i, 2*np.pi)
        omega = np.mod(omega, 2*np.pi)
        RAAN  = np.mod(RAAN, 2*np.pi)
        M     = np.mod(M, 2*np.pi)
    state = np.array([a,e,i,omega,RAAN,M], dtype=np.float64)
    return state
def kep2cart_dep(state, deg=True):
    # https://web.archive.org/web/20160418175843/https://ccar.colorado.edu/asen5070/handouts/cart2kep2002.pdf
    # https://space.stackexchange.com/questions/19322/converting-orbital-elements-to-cartesian-state-vectors
    if len(state) == 7:
        state = state[1:]
    
    mu = MU

    try:
        a,e,i,omega_AP,omega_LAN, M, BSTAR = state
    except ValueError:
        a,e,i,omega_AP,omega_LAN, M = state
        BSTAR = 0
    if deg:
        i         = np.deg2rad(i)
        omega_AP  = np.deg2rad(omega_AP)
        omega_LAN = np.deg2rad(omega_LAN)
        M         = np.deg2rad(M)

    EA = M
    EA_prev = 0
    for n in range(100):
        if n == 0:
            continue
        
        EA += jv(n, n*e)*np.sin(n*M)
        if np.abs(EA - EA_prev) < 1e-12:
            break
        EA_prev = EA

    nu = 2*np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(EA/2))
    #4
    r = a*(1 - e*np.cos(EA))
    #5
    h = np.sqrt(mu*a * (1 - e**2))
    #6
    Om = omega_LAN
    w =  omega_AP

    X = r*(np.cos(Om)*np.cos(w+nu) - np.sin(Om)*np.sin(w+nu)*np.cos(i))
    Y = r*(np.sin(Om)*np.cos(w+nu) + np.cos(Om)*np.sin(w+nu)*np.cos(i))
    Z = r*(np.sin(i)*np.sin(w+nu))

    #7
    p = a*(1-e**2)

    V_X = (X*h*e/(r*p))*np.sin(nu) - (h/r)*(np.cos(Om)*np.sin(w+nu) + np.sin(Om)*np.cos(w+nu)*np.cos(i))
    V_Y = (Y*h*e/(r*p))*np.sin(nu) - (h/r)*(np.sin(Om)*np.sin(w+nu) - np.cos(Om)*np.cos(w+nu)*np.cos(i))
    V_Z = (Z*h*e/(r*p))*np.sin(nu) + (h/r)*(np.cos(w+nu)*np.sin(i))

    state = np.array([X,Y,Z,V_X,V_Y,V_Z], dtype=np.float64)
    return state
