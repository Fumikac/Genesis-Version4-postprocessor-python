import numpy as np
from functions import undulator
from functions import electron as e

def lambda_shift_seed(E_mev, dEds, L):
    '''
    Normalized shift of wavelength at the output of undulator. Used the approximation.
    alpha << 1, lam_in/lam_u << 1
    E_mev: energy of the e-beam at the entrance of the undulator, resonant with seed beam
    dEds: energy chirp of the beam [gamma /m]
    L: propagation distance [m]
    
    '''

    gamma_in= e.gamma(E_mev)

    v = undulator.visa(lambda_s = 421e-9)
    lam_in = v.lambda_1( E_mev)

    alpha = dEds*lam_in/gamma_in #normalized slope

    gamma_hp = gamma_in*(1+alpha/2) #gamma of electrons half a wavelength ahead
    gamma_hn = gamma_in*(1-alpha/2) #gamma of electrons half a wavelength behind
    
    #normalized wavelength shift
    #dlam_lam =  L*(v.v_z(gamma_hp) - v.v_z(gamma_hn))/v.v_z(gamma_in)/lam_in #exact value
    dlam_lam = 2*L*alpha/v.lambda_u #Approximated
    lam_out = lam_in + dlam_lam*lam_in
        
    return lam_in, lam_out
 
def eV_shift_1st(E_mev,lam_seed, dEds, L):
    '''This function not the right soluation. It is a linear approximation.
    when lambda -> eV, this included only 1st order Taylor expansion'''

    c = 299792458
    h_ev_s = 4.135668e-15

    #get radiation photon energy from electron with energy E_mev
    v = undulator.visa(lambda_s = lam_seed)
    lam_in = v.lambda_1(E_mev)
    eV_in = h_ev_s*c/lam_in

    #electron energy
    gamma_in= e.gamma(E_mev)

    #frequency shift
    eV_diff = - 4*h_ev_s*c*dEds*lam_seed*gamma_in*L/v.lambda_u**2/(1+v.k**2/2)

    return eV_in, eV_in + eV_diff

def eV_shift(E_mev,lam_seed, dEds, L):
    
    c = 299792458
    h_ev_s = 4.135668e-15

    #get radiation photon energy from electron with energy E_mev
    v = undulator.visa(lambda_s = lam_seed)
    lam_in = v.lambda_1(E_mev)
    eV_in = h_ev_s*c/lam_in
    
    #electron energy
    gamma_in= e.gamma(E_mev)
    eV_out = h_ev_s*c/lam_in/(1+2*L*dEds*lam_seed/gamma_in/v.lambda_u)
    return eV_in, eV_out
        
def eV_shift_line(E_mev,lam_seed, dEds, L, linear = False):
    '''
    linear: linear approximation, took 1st order of the Taylor expansion. 1/(1+x) ~ 1-x
    '''
    #make a line
    y_line = np.linspace(0, L, 100) #undulator position
    if linear:
        eV_in, x_line = eV_shift_1st(E_mev,lam_seed, dEds, y_line)
    else:
        eV_in, x_line = eV_shift(E_mev,lam_seed, dEds, y_line)
    return x_line, y_line
    


