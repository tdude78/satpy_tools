import os
import sys
from math import exp, sqrt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.interpolate import PchipInterpolator

try:
    from satpy_tools.standard_atm import T_1976_K, h_1976_km, rhos_1976
except ImportError:
    from standard_atm import T_1976_K, h_1976_km, rhos_1976

rho_vals_interp = PchipInterpolator(h_1976_km, np.log(rhos_1976), extrapolate=False)
Ts_interp       = PchipInterpolator(h_1976_km, T_1976_K, extrapolate=False)

# https://github.com/VictorAlulema/1976-Standard-Atmosphere/tree/master

class US1976:
    """
    1976 Standard Atmosphere:
    z: [m]
    Reference:
    http://www.dept.aoe.vt.edu/~mason/Mason_f/stdatm.f
    """

    @staticmethod
    def _std_atmos_model(z):
        """1976 Standard Atmosphere model"""
        K = 34.163195          # Constant
        C1 = .001               # Factor: m to Km
        H = C1 * z / (1 + C1 * z / 6356.766)
        if H < 11:
            T = 288.15 - 6.5 * H
            P = (288.15 / T) ** (- K / 6.5)
        elif H < 20:
            T = 216.65
            P = 0.22336 * exp(- K * (H - 11) / 216.65)
        elif H < 32:
            T = 216.65 + (H - 20)
            P = 0.054032 * (216.65 / T) ** K
        elif H < 47:
            T = 228.65 + 2.8 * (H - 32)
            P = .0085666 * (228.65 / T) ** (K / 2.8)
        elif H < 51:
            T = 270.65
            P = .0010945 * exp(- K * (H - 47) / 270.65)

        elif H < 71:
            T = 270.65 - 2.8 * (H - 51)
            P = .00066063 * (270.65 / T) ** (- K / 2.8)
        elif H < 84.852:
            T = 214.65 - 2 * (H - 71)
            P = 3.9046e-5 * (214.65 / T) ** (- K / 2)
        else:
            # error = 'z:{} [m] ouf of limits for S.A.'.format(z)
            # raise AboveAltitude(error)

            # Extrapolation
            T = Ts_interp(H)
            P = exp(rho_vals_interp(H))
        return T, P

    def temperature(self, zs):
        """Temperature at "z" altitude, km """
        zs = np.atleast_1d(zs)
        zs = zs * 1000
        Ts = np.zeros(zs.shape)
        for i, z in enumerate(zs):
            if z < 1e-3:
                z = 1e-3
            T,_     = self._std_atmos_model(z)
            Tsl     = 288.15                 # Temp. sea level
            T_ratio = T / 288.15
            Ts[i]   = Tsl * T_ratio
        Ts = np.squeeze(Ts)
        return Ts

    def pressure(self, zs):
        """Pressure at "z" altitude, km """
        zs = np.atleast_1d(zs)
        zs = zs * 1000
        Ps = np.zeros(zs.shape)
        for i, z in enumerate(zs):
            if z < 1e-3:
                z = 1e-3
            _,P = self._std_atmos_model(z)
            Psl = 101325                 # Press. sea level
            Ps[i] = Psl * P
        Ps = np.squeeze(Ps)
        return Ps

    def density(self, zs):
        """Density at "z" altitude, km """
        zs = np.atleast_1d(zs)
        zs = zs * 1000
        rhos = np.zeros(zs.shape)
        for i, z in enumerate(zs):
            if z < 1e-3:
                z = 1e-3
            T,P = self._std_atmos_model(z)
            Rsl = 1.225                  # Density sea level
            R_ratio = P / (T / 288.15)
            rho = Rsl * R_ratio
            rhos[i] = rho
        rhos = np.squeeze(rhos)
        return rhos

    def DEPRICATED_sound_speed(self, z):
        """Sound speed at "z" altitude, km """
        z = z * 1000
        T,_ = self._std_atmos_model(z)
        Asl = 340.294                # Sound speed sea level
        T_ratio = T / 288.15
        return Asl * sqrt(T_ratio)

    def DEPRICATED_dynamic_pressure(self, U, z):
        """
        Dynamic pressure at "z" altitude, km
        U: air speed [m/s]
        """
        z = z * 1000
        rho = self.density(self, z)
        return (rho * U ** 2) / 2

    def DEPRICATED_viscosity_dynamic(self, z):
        """Dynamic viscosity - Sutherland Equation"""
        z = z * 1000
        T,_ = self._std_atmos_model(z)
        BT  = 1.458E-06              # Beta constant for viscosity eq.
        return BT * T ** 1.5 / (T + 110.4)

    def DEPRICATED_viscosity_kinematic(self, z):
        z = z * 1000
        u = self.DEPRICATED_viscosity_dynamic(z)
        rho = self.density(z)
        return u / rho

    def DEPRICATED_Reynolds(self, x, U, z):
        """Reynolds number
           U: air speed [m/s]
           x: reference length [m] (airfoil/wing chord)
        """
        z = z * 1000
        rho = self.density(z)
        u   = self.DEPRICATED_viscosity_dynamic(z)
        return (rho * U * x) / u

    def DEPRICATED_Mach(self, U, z):
        z = z * 1000
        A = self.DEPRICATED_sound_speed(z)
        return U / A



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    atmos = US1976()
    alt   = 100
    alt_array = np.linspace(0, 100, 100)
    print(atmos.temperature(alt))
    print(atmos.pressure(alt))
    print(atmos.density(alt))

    plt.figure()
    plt.plot(atmos.temperature(alt_array), alt_array)
    plt.xlabel('Temperature [K]')
    plt.ylabel('Altitude [km]')
    plt.grid()

    plt.figure()
    plt.plot(atmos.pressure(alt_array), alt_array)
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('Altitude [km]')
    plt.grid()

    plt.figure()
    plt.plot(atmos.density(alt_array), alt_array)
    plt.xlabel('Density [kg/m^3]')
    plt.ylabel('Altitude [km]')
    plt.grid()

    plt.show()
    
