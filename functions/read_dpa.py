"""This is for old versions of Genesis 1.3 (not Version 4). It can read particle files .dpa and .par files, and analyze the particle distributions.
"""

import time
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import mean, std, inf, shape, append, complex128, complex64
from matplotlib.animation import FuncAnimation
from functions.read_beamfile import ebeam
from functions.plot_func import*
import copy


class GenesisParticlesDump:
    '''
    Genesis 1.3 (version1) particle *.dpa or *.par files storage object
    .dpa store the particle parameters at the exit of the undulaotr. .par saves data at all positions throught the undulator (1 dimension hiher than .dpa).
    '''

    def __init__(self):
        self.e = [] #energy
        self.ph = [] #particle phase
        self.x = [] #position x
        self.y = [] #position y
        self.px = [] # moment x (normalized to mc)
        self.py = [] # momen y (normalized to mc)

        self.zeta = []
        self.nslice = []
        self.nz = []
        self.npart = []
        self.xlamds = []
        self.smin = []
        self.smax = []    

        self.dirPath = ''
    
    def save_par_file(self, fprefix=None, smin=0, smax=None):
        '''
        Save a particle file in .npz format
        (smin, smax): Min and Max slice number to save. (possible up to NSLICE) All slices will be saved if not specified.
        fprefix: name of the file. ex. 'par14'
        '''

        if smax is None:
            smax = self.nslice

        par = GenesisParticlesDump()
        par.e = self.e[smin:smax,:,:]
        par.ph = self.ph[smin:smax,:,:]
        par.x = self.x[smin:smax,:,:]
        par.y = self.y[smin:smax,:,:]
        par.px = self.px[smin:smax,:,:]
        par.py = self.py[smin:smax,:,:]

        #longitudinal bunch coordinate
        par.zeta = np.zeros(par.ph.shape)
        for i in range(smax-smin):
            #position in light frame including phase
            par.zeta[i,:,:] = (par.ph[i, :, :]/2/np.pi+i+smin)*self.xlamds

        if not fprefix:
            fprefix = 'par'
        if not smax:
            fname = fprefix+'.npz'
        else:
            fname = fprefix+'_s'+str(smin)+'_'+str(smax)+'.npz'
        np.savez(fname, e=par.e, ph=par.ph, x=par.x, px=par.px, y=par.y, py=par.py, zeta=par.zeta,
                 nslice=smax-smin, nz=self.nz, npart=self.npart, xlamds=self.xlamds, smin=smin, smax=smax, dirPath = self.dirPath)
        print(fname+' saved')

        return fname

    def plot_beam(self, var, xvar ='zeta', z_i=-1, markersize=0.1, xlim= None, ylim=None):
        '''
        Plot particle variable 'var' vs 'xvar'
        var: 'e', 'ph', 'x', 'y', 'px', 'py'
        z_i: beam position. Choose any int below nz. -1: undulator exit
        npart: # of particles. NPART in genesis.
        '''

        #Get parameters of a whole beam
        ydata = getattr(self, var)
        xdata = getattr(self, xvar)
        
        if ydata.ndim == 2: #for .dpa file
            data_snap = ydata
            xdata_snap = xdata
        else: #for .par file
            data_snap = ydata[:,z_i,:]
            xdata_snap = xdata[:,z_i,:]
        #Get parameters of a whole beam
        data_z = data_snap.reshape(-1,) #chande to 1D
        xdata_z = xdata_snap.reshape(-1,)

        #plt.plot(s_z*1e6,data_z,'.', markersize=markersize)
        plt.scatter(xdata_z, data_z, s=markersize)
        if xlim != None:
            plt.xlim(xlim)
        if ylim != None:
            plt.ylim(ylim)
        plt.xlabel(xvar)
        plt.ylabel(var)
        
        return None
    
    
    def peak_current(self):
        '''
        get z position where the input ebeam current is at the peak
        Use ebeam class from read_beamfile.py
        return: z_maxcur, i_maxcure: z position and index where current is maximum
        '''
        beam = ebeam(str(self.dirPath) + '/workfile')
        return beam.peak_current()
    
    def beam_line_density(self, file=None):
        '''
        Get line density of the beam. Can be used to weight the histogram of particle distribution.
        :return: Linear density (Density*Transversebeamarea) for each slice. size: (smax-smin+1,)
        '''
        beam = ebeam(str(self.dirPath) + '/workfile')
        weights = beam.line_density()
        weights = weights.values[self.smin:self.smax]
        return weights
    
    def par_z_hist(self, nz_i=-1, isweight = True, roi=0):
        '''
        Get line density of the beam. [# electrons/m] 
        nz_i: time slice. -1 for undulator exit, 0 for undulator entrance. max self.nz.
        roi: if >0, outside of roi around the current peak will be zeroed (zero padding) 
        '''
        s_z_0 = self.zeta[:,0,:].reshape(-1,) #particle positions at the entrance
        s_z = self.zeta[:,nz_i,:].reshape(-1,)#Make 1D z pos. data
        if isweight:
            weights = self.beam_line_density()
            weights_z = np.repeat(weights, self.npart) #same size as s_z
        else:
            weights_z = None
        nbin = 2**12+1 #Want the output of hist to be power of 2 for TTF
        x, y = get_hist(s_z, min(s_z_0), max(s_z_0), weight=weights_z,N=nbin, density=False)
        #normalize the distribution (density distribution), but keep the weightss
        const = self.nslice*self.npart/nbin
        y = y/const
        
        
        #ROI around peak current
        if roi > 0:
            _, z_maxcur = self.peak_current()
            x_trim = copy.deepcopy(x)
            y_trim = np.zeros(y.shape)
            x_trim[abs(x-z_maxcur) <= roi/2.], y_trim[abs(x-z_maxcur) <= roi/2.] = x[abs(x-z_maxcur) <= roi/2.], y[abs(x-z_maxcur) <= roi/2.]
            return x_trim, y_trim
        else:
            return x, y
       
    def get_par_z_fft_2d(self, nz = None, l_u = 0.018,isweight=True, roi = 0):
        '''
        Plot FFT of particle z positions along the undulator position in 2D.
        number of undulator periods to plot. Max self.nz (exit of undulator)
        l_u: undualtor period
        isweight: if you weight your particles to have accurate destribution. if False, particles are evenly destributed over z (raw output of genesis).
        roi: if you want to only FFT the region around the peak of the beam current. beam length in [m].
        '''
        if not nz:
            nz = self.nz
        x1, y1 = self.par_z_hist(0, isweight=isweight, roi = roi)
        N = len(x1)+1

        C = np.zeros((N//2, nz)) #2D array for mapping

        for i in range(nz):
            x1, y1 = self.par_z_hist(i, isweight=isweight, roi = roi)
            xf, C[:,i] = get_fft(x1, y1)

        yf = np.arange(nz)*l_u
        h = 4.135667696e-15 #[eV*s]
        return h*xf[5:], yf, np.log10(C[5:,:])
    
        
    

def read_par_file(filePath, nslice=None, nwig=None, npart=None, debug=1, xlamds = 4.21e-7):
    '''
    Read genesis particle file *.par
    param
    filePath: full path of .par file
    nslice: number of slices.(int) max possible is NSLICE in genesis
    nwig: # of positions. NWIG in genesis
    npart: # of particles. NPART in genesis
    debug: ON(1) or OFF(0)
    xlamds: distance between slices. XLAMDS in genesis.
    
    :return: class par with 6 parameters and others
    '''
    assert npart != None, 'number of particles per bin is not defined'
    npara = 6 # of parameters in .par file
    nbins = 4 # of bins in .par (only 1 of these have values)
    nz = nwig - 1
    count = nz*npart*nslice*npara*nbins


    if not os.path.isfile(filePath):
        raise IOError('      ! par file ' + filePath + ' not found !')
    else:
        start_time = time.time()
        print('    reading particle file')
        b_raw = np.fromfile(filePath, dtype=float, count=count)

    assert npart != None, 'number of particles per bin is not defined'
    assert len(b_raw) == count, 'file size does not match with input numbers'

    b = b_raw.reshape(nslice, nz, npara, nbins, npart)
    
    par = GenesisParticlesDump()
        
    par.e = b[:,:,0,0,:]
    par.ph = b[:,:,1,0,:]
    par.x = b[:,:,2,0,:]
    par.y = b[:,:,3,0,:]
    par.px = b[:,:,4,0,:]
    par.py = b[:,:,5,0,:]
    
    par.nslice = nslice
    par.nz = nz
    par.npart = npart
    par.xlamds = xlamds
    par.smin = 0
    par.smax = nslice
    par.dirPath = os.path.dirname(filePath)

    #longitudinal bunch coordinate
    par.zeta = np.zeros(par.ph.shape)
    for i in range(nslice):
        #position in light frame including phase
        par.zeta[i,:,:] = (par.ph[i, :, :]/2/np.pi+i)*xlamds

    if debug > 0:
        print('    ', nslice, ' slices loaded')
        print('b shape:', b.shape, ' = ', count)
        print('b length:', len(b_raw))
        print('parameter shape:', par.e.shape)
        print('    done in %.2f sec' % (time.time() - start_time))
    
    return par

def read_dpa_file(filePath, npart=None, debug=1, xlamds = 4.21e-7):
    '''
    reads genesis particle dump file *.dpa
    returns GenesisParticlesDump() object
    '''

    if not os.path.isfile(filePath):
        raise IOError('      ! dpa file ' + filePath + ' not found !')
    else:
        start_time = time.time()
        print ('        - reading from ' + filePath)
        b = np.fromfile(filePath, dtype=float)

            
    dpa = GenesisParticlesDump()

    assert npart != None, 'number of particles per bin is not defined'
    npara = 6 # of parameters in .par file
    nbins = 4 # of bins in .par (only 1 of these have values)
    nslice = int(len(b)/npart/npara/nbins)

    b = b.reshape(nslice, npara, nbins, npart)
    dpa.e = b[:,0,0,:]
    dpa.ph = b[:,1,0,:]
    dpa.x = b[:,2,0,:]
    dpa.y = b[:,3,0,:]
    dpa.px = b[:,4,0,:]
    dpa.py = b[:,5,0,:]
    
    dpa.nslice= nslice
    dpa.npart = npart
    dpa.xlamds = xlamds
    dpa.dirPath = os.path.dirname(filePath)
    #all other components are 0
    
    #longitudinal bunch coordinate
    dpa.zeta = np.zeros(dpa.ph.shape)
    for i in range(nslice):
        #position in light frame including phase
        dpa.zeta[i,:] = (dpa.ph[i, :]/2/np.pi+i)*xlamds
        
    if debug > 0:
        print('      done in %.2f sec' % (time.time() - start_time))
        
    del b

    return dpa


def load_par_file(outfile):
    '''
    Load .npz file (saved from .par file)
    '''
    npzfile = np.load(outfile)
    
    par = GenesisParticlesDump()
    par.e = npzfile['e']
    par.ph = npzfile['ph']
    par.x = npzfile['x']
    par.y = npzfile['y']
    par.px = npzfile['px']
    par.py = npzfile['py']
    par.zeta = npzfile['zeta']

    par.nslice = npzfile['nslice']
    par.npart = npzfile['npart']
    par.xlamds = npzfile['xlamds']
    par.smin = npzfile['smin']
    par.smax = npzfile['smax']
    par.dirPath = os.path.dirname(outfile)
    if 'nz' in npzfile.files:
        par.nz = npzfile['nz']

    print('parameter shape:', par.e.shape)

    return par


def main():

    #   Read
    #dpa = read_dpa_file('D:/Server/USER/Fumika/VISA_sample/template.out.dpa', npart = 8192)

    #particle files
    filePath = 'D:/Server/USER/Fumika/Exp3_Data/14/template.out.par'
    nslice = 10
    nwig = 224
    npart = 8192
    
    par = read_par_file(filePath,nslice=nslice, nwig=nwig,npart=npart)
    par.plot_par_z('ph', z_i=150, nslice=nslice)

if __name__ == '__main__':
    main()
