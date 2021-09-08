import numpy as np
import h5py
from functions.plot_func import get_hist
import os

class ParticleDump:
    """
    Read output file of Genesis 1.3 Verison4. This is the main class used for postprocessing. Check result-*.ipynb for example use cases.
    """
    
    def __init__(self, path):
        #load file
        self.filepath = path
        self.filename = self.filepath.split('/')[-1]
        self.file = h5py.File(self.filepath, 'r')
        self.dir = os.path.dirname(self.filepath)
        
        #get constants
        self.beamletsize = self.file['beamletsize'][0]
        self.one4one = self.file['one4one'][0]
        self.refposition = self.file['refposition'][0]
        self.slicecount = self.file['slicecount'][0]
        self.slicelength = self.file['slicelength'][0]
        self.slicespacing = self.file['slicespacing'][0]
        self.n_particles = self.file['slice000001']['x'][()].shape[0]     
                   
        #get zeta position (internal bunch coordinate)
        self.get_variable('theta')
        self.zeta = np.array([(self.theta[i,:]/2/np.pi+i)*self.slicespacing for i in range(self.slicecount)] )
        
        
    def get_data(self):
        """
        get 2D  beam parameters
        """
        for para in ['gamma', 'theta', 'px', 'py', 'x', 'y']:
            if not hasattr(self, para):
                self.get_cariable(para)
               
    def get_variable(self, variable):
        """
        get a variable: 'gamma', 'px', 'py', 'theta', 'x', 'y'
        output: numpy 2d array (slicecount x n_particles)
        """
        #  get parameters of all slices     
        data = np.array([self.file['slice'+f"{i+1:06}"][variable][()] for i in range(self.slicecount)])
        setattr(self, variable, data)
        
    def slice_particle_density(self):
        """Get line density of the beam (a point per slice). Can be used to weight the histogram of particle distribution.
        :return: Linear density (Density*Transverse beam area) size: (slicecount,)
        """
        if not hasattr(self,'gamma'):
            self.get_variable('gamma')
        
        #get current data
        self.current = np.array([self.file['slice'+f"{i+1:06}"]['current'][()][0] for i in range(self.slicecount)]) 
        
        #get weights for the particle distribution 
        gamma_mean = np.array([np.mean(self.gamma[i,:]) for i in range(self.slicecount)])
        c = 299792458
        e = 1.6022e-19
        velocity = c * np.sqrt((gamma_mean ** 2 - 1) / gamma_mean ** 2)
        weights= np.divide(self.current, velocity*e)
        
        return weights
        
    def line_density(self, zeta_min = None, zeta_max = None, weights = None):
        """
        Get 1D line density in size (nbin,). Include data from all particles(#slices x #particles)
        """
        zeta_allslices = self.zeta.reshape(-1) #combine zeta of all slices, 1D 
        
        if not zeta_min:
            zeta_min = zeta_allslices.min()
        if not zeta_max:
            zeta_max = zeta_allslices.max()
        
        #weights can be same for all particle dumps. Just feed the initial weights if you want to reduce the calculation.
        if not weights:
            weights = self.slice_particle_density()
        weights_allslices = np.repeat(weights, self.n_particles)
            
        nbin = 2**11+1 #Want the output of hist to be power of 2 for TTF
        x, y = get_hist(zeta_allslices, xmin = zeta_min, xmax = zeta_max, weights=weights_allslices, N=nbin)
        y = y/(self.slicecount*self.n_particles/nbin) #normalize the distribution
        
        return x, y
        