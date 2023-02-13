import numpy as np
from utils import *

class QUdata():

    def __init__(self, cfg):

        self.cfg = cfg

    def read_data(self):

        # freq       I        Q        U        V        N
        data = np.loadtxt(self.cfg.data_path+self.cfg.data_file)
        self.nu   = data[:,0]*1e6
        self.stokesI  = data[:,1]  # stokes I in file is Russ' model, not the raw data
        self.stokesQn = data[:,2]
        self.stokesUn = data[:,3]
        self.stokesVn = data[:,4]
        Qbkg = data[:,5]
        Ubkg = data[:,6]
        Vbkg = data[:,7]
        self.noise    = data[:,8]

        if self.cfg.bkg_corr:
            self.stokesQn -= Qbkg
            self.stokesUn -= Ubkg
            
        if self.cfg.pol_frac:
            self.stokesQn *= 100./self.stokesI
            self.stokesUn *= 100./self.stokesI
            self.noise    *= 100./self.stokesI
        
        const_c = 3e8

        # make data in lambda^2:
        self.l2 = (const_c/self.nu)**2

        #if self.cfg.plot_raw:
        #    self.plot_rawdata()

        return
        
    def norm_data(self):
        
        self.norm = np.max([np.max(np.abs(self.stokesQn)), np.max(np.abs(self.stokesUn))])
        
        self.stokesQn = (self.stokesQn)/self.norm
        self.stokesUn = (self.stokesUn)/self.norm
        self.noise    = self.noise/self.norm
        
        return

    def unnorm_data(self, inQ, inU, inNoise):
        
        outQ = (inQ*self.norm)
        outU = (inU*self.norm)
        outNoise = inNoise*self.norm
    
        return outQ, outU, outNoise


    def read_cat(self):

        f = open(self.cfg.catpath+self.cfg.catfile)
        cat_cols = f.readline().rstrip("\n")
        cat_unts = f.readline().rstrip("\n")
        cat_data = np.loadtxt(self.cfg.catpath+self.cfg.catfile)
        
        self.cat_data = cat_data

        return
