import numpy as np
from scipy.special import jv
from constants import MU

# 02/07/24

def cart2kep(state, deg=True):
    # https://web.archive.org/web/20160418175843/https://ccar.colorado.edu/asen5070/handouts/cart2kep2002.pdf
    # https://space.stackexchange.com/questions/19322/converting-orbital-elements-to-cartesian-state-vectors
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
    state = np.array([a,e,i,omega,RAAN,M], dtype=np.float64)
    return state


def kep2cart(state, deg=True):
    # https://web.archive.org/web/20160418175843/https://ccar.colorado.edu/asen5070/handouts/cart2kep2002.pdf
    # https://space.stackexchange.com/questions/19322/converting-orbital-elements-to-cartesian-state-vectors
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



def parse_TLE(line1, line2):
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

    result = MJD, np.array([SMA, ECC, INC, RAAN, AOP, MA], dtype=np.float64)

    return result

