import h5py
import numpy as np
from IPython.display import HTML, display
import tabulate

class GenesisOut:

    def __init__(self, path):
        #load file
        self.filepath = path
        self.filename = self.filepath.split('/')[-1]
        self.file = h5py.File(self.filepath, 'r')
        self.dict = {group:list(self.file[group].keys()) for group in self.file}  
        
        #constants
        speed_of_light = 299792458.0 #m/s
        h_eV_s = 4.135667516e-15    #eV s

        #generate attributes
        self.name_douplicate = ['xsize', 'ysize', 'xposition', 'yposition']
        self.group_douplicate = ['Field', 'Beam']        
        self.generate_attributes()
        
        #central frequency of the radiation
        self.lambda_rad =  self.lambda_rad()
        
        #axis
        self.s = self.s()
        self.t = self.s/speed_of_light #time
        self.dt = self('sample') * self('lambdaref')/ speed_of_light
        
        #photon energy/spectrum
        self.spec_i_near = self.spec_i('nearfield') #spectrum intensity
        self.spec_p_near = self.spec_i_near**2 #spectrum power (magnitude squared)
        self.spec_i_far = self.spec_i('farfield') #spectrum intensity
        self.spec_p_far = self.spec_i_far**2 #spectrum power (magnitude squared)
        
        #axis for spectrum
        self.e0 = h_eV_s * speed_of_light / self.lambda_rad #resonant [eV]
        self.freq_ev = np.fft.fftshift(h_eV_s * np.fft.fftfreq(len(self.s), self.dt) + self.e0, axes=0)
        self.freq_lamd = h_eV_s * speed_of_light * 1e9 / self.freq_ev

        #maximum power along z
        self.power_z = np.max(self.power, 1)
        
        #display
        self.show_contents()
        
    def __call__(self, *name):
        """
        Able to call self(name)
        (parent, child), or just (child), or (parent_child)
        """    
        
        #when only one input argument
        if len(name)==1:
            '''Find the parent name'''
            
            #reject the douplicate names
            if name[0] in self.name_douplicate:
                print('Please put (group, parameter)')
                return None
          
            for group in self.file:
                for param in self.file[group]:
                    if name[0] == param:
                        (parent, child) = (group, name[0])
        #two input arguments     
        elif len(name)==2:
            (parent, child) = name
            
        if child == 'Version':
                return {key:self.file['Meta']['Version'][key][()] for key in self.file['Meta']['Version'].keys()}
        else:
            parameter = self.file[parent][child][()]
            if parameter.shape == (1,):
                return parameter[0]
            elif parameter.shape ==(1,1):
                return parameter[0][0]
            else:
                return parameter
            
    def generate_attributes(self):
        """
        generate attributes in initial setting
        """
        for group in self.dict:
            for param in self.dict[group]:
                if group in self.group_douplicate and param in self.name_douplicate:
                    setattr(self, group+'_'+param, self(group, param))
                else:
                    setattr(self, param, self(group, param))
        
    def s(self):
        """
        get s axis
        """
        (dim_z, dim_s) = self('power').shape
        ds = self('sample')
        sref = self('lambdaref')
        return np.arange(dim_s) * ds *sref
    
    def lambda_rad(self):
        """
        returns central frequency of the radiation mode, which is lambda in the field section of the input file if specified, or lambda0(lambdaref) if not specified.
        """
        InputFile = self('Meta','InputFile').decode("utf-8")
        d_InputFile = dict([item.replace(' ','').split('=') for item in InputFile.splitlines() if '=' in item])
        if 'lambda' in d_InputFile:
            return float(d_InputFile['lambda'])
        else:
            return self.lambdaref
            
    def spec_i(self, near_or_far = 'nearfield'):
        """
        get spectrum intensity
        """
        intensity = self('intensity-'+near_or_far)
        phase = self('phase-'+near_or_far)

        spec_i = abs(np.fft.fft(np.sqrt(np.array(intensity)) * np.exp(1.j * np.array(phase)), axis=1))
        return np.fft.fftshift(spec_i, axes=1) 

    def show_info(self, group):
        """
        Show the size of all the keys, and the value if the size is 1
        """
        print(group)
        for child in self.dict[group]:
            if child=='Version':
                print(child, ':', self(group, child))
            elif not self(group, child).shape:
                print(child, ':',self(group, child))
            else:
                print(child,': size', self(group, child).shape)

    def show_contents(self):
        """
        Show all the contents
        """
        print(self.filename, 'loaded')

        table = [['group', 'parameter']]
        for group in self.file:
            table.append([group, self.dict[group]])
        display(HTML(tabulate.tabulate(table, tablefmt='html')))

        print('Call directly as an attribute or call (parameter) or (group, parameter) to retrieve data')
        print('Use .show_info(group) to show parameter shapes')

    