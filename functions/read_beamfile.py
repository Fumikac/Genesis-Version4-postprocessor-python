'''read beam file, which is usually called workfile in my folders
beam parameters (http://genesis.web.psi.ch/Manual/parameter_beam.html)
'''
import pandas as pd
import numpy as np
import os


class ebeam:
    '''
    Deal with beamfile made by elegant.
    '''

    def __init__(self, filePath):
        self.file = filePath
        self.dir = os.path.dirname(os.path.abspath(self.file)) + '/'

        #data
        self.header = None
        self.data = None
        self.n_slice = None

        #load beam file
        self.load_ebeamfile()

    def load_ebeamfile(self):
        '''Load ebeam file, return a Dataframe'''
        f = open(self.file, 'r')
        line1=f.readline()
        line2=f.readline()
        line3 = f.readline()  # read the third line first to get names for columns
        self.header = line1+line2+line3
        self.data = pd.read_csv(self.file, sep='\s+', names=line3.replace(' ? COLUMNS ', '').split(), skiprows=3)
        f.close()

        self.n_slice = len(self.data.index)

        return None
    
    def i_Imax(self):
        '''Index of max current position'''
        return np.argmax(np.array(self.data['CURPEAK']))

    def peak_current(self):
        '''
        get z position where the input ebeam current is at the peak
        z_maxcur: z position where current is maximum
        i_maxcur: index where current is maximum
        '''       
        #shift to make a tail position zero
        zpos = self.data['ZPOS'] - np.min(self.data['ZPOS'])

        # get z position of the current peak
        return self.data['CURPEAK'][self.i_Imax()], zpos[self.i_Imax()]
    
    
    def peak_beamsize(self):
        '''
        get transverse RMS beam size in x and y
        '''
        sigma_x_peak = self.data['RXBEAM'][self.i_Imax()]
        sigma_y_peak = self.data['RYBEAM'][self.i_Imax()]
        return sigma_x_peak, sigma_y_peak

    def line_density(self):
        ''''Get line density of the beam. Can be used to weight the histogram of particle distribution.
        :return: Linear density (Density*Transversebeamarea) for each slice. size: (nslice,)
        '''
        cur = self.data['CURPEAK']
        gamma = self.data['GAMMA0']
        c = 299792458
        e = 1.6022e-19
        vel = c * np.sqrt((gamma ** 2 - 1) / gamma ** 2)
        weights= np.divide(cur, vel*e)
        return weights

    def beamfile_row(self,i, j, cur_ratio=1):
        '''
        Row of txt to write in beamfile
        :param i: index of z position
        :param j: index of all other paramters
        :param cur_ratio: Current you want to devide with
        :return: row of txt
        '''
        data = self.data
        row_txt = '%.11e %.9f %.13f %.11e %.11e %.15f %.15f %.11e %.11e %.11e %.11e %.10f %.10f %.12e %.1f\n' % (
            data['ZPOS'][i], data['GAMMA0'][j], data['DELGAM'][j], data['EMITX'][j], data['EMITY'][j],
            data['RXBEAM'][j], data['RYBEAM'][j], data['XBEAM'][j], data['YBEAM'][j], data['PXBEAM'][j],
            data['PYBEAM'][j], data['ALPHAX'][j], data['ALPHAY'][j], data['CURPEAK'][j] * cur_ratio, data['ELOSS'][j])
        return row_txt

    def save_beamfile(self, savefile, cur_ratio = 1, reverse = False):
        '''
        Save beam file
        :param reverse: Boolean. Reverse the current (reverse chirp)
        '''
        fw = open(savefile, 'w')
        fw.write(self.header)

        for i in range(self.n_slice):
            if reverse:
                fw.write(self.beamfile_row(i, self.n_slice - 1 - i, cur_ratio))
            else:
                fw.write(self.beamfile_row(i, i, cur_ratio))

        fw.close()



