import numpy as np

def gamma(E):
    '''Return relativofistic gamma. Takes electron energy [MeV] '''
    E_rest = 0.510623989196873 #MeV
    return E/E_rest

def E(gamma):
    '''Returns energy in MeV. Takes relativistic gamma.'''
    E_rest = 0.510623989196873 #MeV
    return gamma * E_rest

def vel(gamma):
    '''return electron velocity (free particle) [v/s]
    gamma: electron energy
    '''
    c = 299792458
    return c * np.sqrt((gamma ** 2 - 1) / gamma ** 2)