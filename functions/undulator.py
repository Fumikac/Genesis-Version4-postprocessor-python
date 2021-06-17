import numpy as np
import functions.electron as electron
import scipy
import scipy.special

class visa:

    def __init__(self, lambda_s = None):
        '''
        l_s: fundamental wavelength
        '''
        self.lambda_u = 0.018 #18mm
        self.k = 1.26 #(RMS aw = 0.89 is used for Genesis simulation. aw^2 = k^2/2)
        self.B = 0.75 #peak b field[T]
        self.L = 4 #undulator length
        
        if lambda_s:
            self.lambda_s = lambda_s #resonant wavelength
            self.gamma_s = self.gamma_res(lambda_s) #electron energy(gamma) resonant with l_s
            self.E_s = electron.E(self.gamma_s) #Energy(MeV) of the electron resonant with l_s
            self.v_z_s = self.v_z(self.gamma_s) #longitudinal velocity of the electron resonant with l_s

    
    def gamma_res(self, lambda_s):
        '''Takes fundamental wavelength [m], and returns electron energy(gamma) resonant with the radiation wavelength.
        '''
        return  np.sqrt((1 + self.k ** 2/2) * self.lambda_u / (2 * lambda_s))
    
    def v_z(self, gamma):
        '''Return electron longitudinal velocity in the undulator. first order.
        gamma: electron energy
        '''
        c = 299792458
        return (1-1/(2*gamma**2)*(1+self.k**2/2))*c
    
    def lambda_1(self, energy):
        '''Takes energy  electron beam [MeV], and returns fundamental wavelength of the radiation.'''
        return self.lambda_u / 2 / (electron.gamma(energy) ** 2) * (1 + self.k ** 2 / 2)
        
    def K_mod(self):
         '''Return the modified undulator parameter, which includes the longitudinal oscillation'''
         
         x_J=self.k**2/(4+2*self.k**2)
         JJ = scipy.special.j0(x_J) - scipy.special.jv(1, x_J)
         return self.k*JJ

    def K(self):
        #Return undulator parameter K. Takes magnetic field [T]
        return 0.934*self.lambda_u*100*self.B

    def rho(self, gamma, I, sigma_x):
        '''FEL parameter rho
        gamma: electron energy
        I: e-beam peak current [A]
        sigma_x: RMS transverse beam size

        %VISA:0.0017
        %LEUTL:0.0044
        %FLASH:0.0017
        %LCLS:5.8467e-4
        %SACLA:2.8725e-04
        %FERMI:0.0011
        '''
        lambda_1 = self.lambda_1(gamma)
        I_A = 17045
        x_J=self.k**2/(4+2*self.k**2)
        JJ = scipy.special.j0(x_J) - scipy.special.jv(1, x_J)
        rho = ( I/(8*np.pi*I_A)*(self.k*JJ/(1+self.k**2/2))**2*gamma*lambda_1**2/(2*np.pi*sigma_x**2))**(1/3)
        return rho
    