import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

def plot_2d(x,y,C):
    
    dx, dy = x[-1]-x[-2], y[-1]-y[-2]
    x = np.append(x, x[-1]+dx)
    y = np.append(y, y[-1]+dy)
    nx, ny = C.shape
    X = np.tile(x, (ny+1,1))
    Y = np.tile(y, (nx+1,1)).transpose()
    plt.pcolormesh(X, Y, C.transpose())
    
    return None

def get_hist(s_z, xmin = None, xmax=None, N=5000, weights=None):
    '''
    Get a histogram from a 1D scatter data
    N: number of sample points
    
    output: probability density
    '''
    if xmin is None and xmax is None:
        xmin = min(s_z)
        xmax = max(s_z)
        
    T = (xmax - xmin)/N #sample spacing
    y, bins = np.histogram(s_z, bins=np.linspace(xmin, xmax, N), weights=weights)
    x = bins[:-1]-T/2
    
    return x,y

def get_fft(x, y, n=None):
    '''
    Get FFT of signal y(x).
    (x,y): signal (both 1D). x has to be evenly spaced.
    n: if n > y.shape, y is zero padded.
    '''
    c = 299792458
    T = (x[1] - x[0])/c #sample spacing
    N = len(y) + 1 #number of sample points
    
    #FFT
    yf = scipy.fftpack.fft(y, n)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    I = 2.0/N*np.abs(yf[:N//2])

    return xf, I

def linear_line(dy, x0, x1):
    '''
    Get y (1D array) of a linear plot

    dy: y1-y0. range of y
    x0, x1: starting and ending x point of the line
    return: 1D array of x and y
    '''
    x_line = np.linspace(x0, x1, 100)
    y_line = dy / (x1 - x0) * (x_line - x0)
    return x_line, y_line

