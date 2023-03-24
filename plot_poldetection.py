#! /usr/bin/env python

import pylab as pl
from scipy.optimize import minimize
from scipy import stats
from scipy.ndimage import median_filter
from scipy.interpolate import UnivariateSpline

from utils import *
from dataio import *
from faraday_stuff import *

# ----------------------------------------------------------
# functions to bin up polarisation fraction as a function of lambda-squared:

def make_bins(x,y):

    min_counts = 10
    bnum = np.zeros_like(x)
    max_dx = np.abs(x[-1] - x[-2])*2

    bin = 0
    bnum[0] = 0
    bin_count = 1
    for i in range(1,len(x)-1):
        dx1 = x[i] - x[i-1]
        dx2 = x[i+1] - x[i]
        if (bin_count<10):
            if (dx1<=max_dx):
                bnum[i] = bin
                bin_count+=1
            else:
                bin+= 1
                bin_count=0
                bnum[i] = bin
                bin_count+=1
        elif (bin_count>=10):
            if (dx2<=max_dx):
                bin+=1
                bin_count = 0
                bnum[i] = bin
                bin_count+=1
            else:
                bnum[i] = bin
                bin_count+=1
                
    bnum[-1] = bnum[-2]
            
    nbins = np.max(bnum)

#    for i in range(len(bnum)):
#        print(i, x[i], bnum[i])
    
    return bnum

# ----------------------------------------------------------
# functions to fit polarisation angle as a function of lambda-squared:

def line(p,x):
    m, c = p
    return m*x + c

# must be a more pythonic way to do this...
def dtheta(a,b):
    x1 = np.abs(a-b)
    x2 = 2*np.pi - np.abs(a-b)
    x = np.vstack((x1,x2))
    return np.amin(x, axis=0)
    
def fit_chi(x,y,err,p0):

    nll = lambda *args: np.sum(dtheta(y,line(*args))**2/err**2)
    
    initial = p0 + 0.01 * np.random.randn(2)
    bnds = ((None, None), (-np.pi, np.pi))
    
    soln = minimize(nll, initial, bounds=bnds, args=(x))
    p = soln.x

    return p


def spline_interpolate(x, y, window=3000):
    """
    Calculate a spline fit for the input 1D array.

    Window is an arbitrary smoothing factor, a small number makes the spline
    follow the scatter very closely, too large a number forces it to be nearly
    linear.

    Inputs:
    y           1D array, np.array[float]
    win         Window size, int

    Returns:
    filter_y    Filtered input
    """

    spl = UnivariateSpline(x, y, s=window*x.size)

    xinterp = np.linspace(x.min(), x.max(), 1000)
    return xinterp, spl(xinterp)


# ----------------------------------------------------------
# ----------------------------------------------------------

cfg = QUcfg()              # init config class

vars = cfg.parse_args()
cfg_file = vars['config']
cfg.read_cfg(cfg_file)     # read config file

if vars['srcid'] is not None:
    cfg.data_file = 'ID{}_polspec.txt'.format(vars['srcid'])

data = QUdata(cfg)    # init data class
data.cfg.pol_frac = False
data.cfg.bkg_correction = False

data.read_data()           # read QU data
data.read_cat()            # read catalogue file (if it exists)

# ----------------------------------------------------------
# Make panel plot:

pl.rcParams['figure.figsize'] = [10, 10]
pl.rcParams['figure.dpi'] = 150

fig = pl.figure()

# ----------
# Q vs U

ax1 = fig.add_axes([0.1, 0.1, 0.5, 0.5])

ax1.errorbar(data.stokesQn, data.stokesUn, xerr=data.noise, yerr=data.noise, fmt='', lw=0.1, ls='')
sc=ax1.scatter(data.stokesQn, data.stokesUn, c=data.l2, s=20)
ax1.axhline(y=0, ls=':', c='lightgray')
ax1.axvline(x=0, ls=':', c='lightgray')
ax1.set_ylabel(r"Q [$\mu$Jy beam$^{-1}$]", fontsize=12)
ax1.set_xlabel(r"U [$\mu$Jy beam$^{-1}$]", fontsize=12)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cbaxes = inset_axes(ax1, width="70%", height="5%", loc=1)
pl.colorbar(sc, cax=cbaxes, orientation='horizontal', label=r"$\lambda^2$ [m$^2$]")

# ----------
# Q, U, P

ax2 = fig.add_axes([0.7, 0.1, 0.5, 0.5], xlim=(np.min(data.l2),np.max(data.l2)))

p_mag = np.sqrt(data.stokesQn**2 + data.stokesUn**2)

xpfilt, p_mag_filt = spline_interpolate(data.l2[::-1][1:], p_mag[::-1][1:], window=500)
xqfilt, stokesQ_filt = spline_interpolate(data.l2[::-1][1:], data.stokesQn[::-1][1:], window=500)
xufilt, stokesU_filt = spline_interpolate(data.l2[::-1][1:], data.stokesUn[::-1][1:], window=500)

ax2.errorbar(data.l2[::-1],p_mag[::-1], yerr=data.noise, fmt='.', c='black', capthick=0, lw=0.1, label='P')
ax2.plot(xpfilt,p_mag_filt, c='black', lw=1)

ax2.errorbar(data.l2[::-1],data.stokesQn[::-1], yerr=data.noise, fmt='s', ms=3, c='blue', capthick=0, lw=0.1, label='Q')
ax2.plot(xqfilt,stokesQ_filt, c='blue', lw=1)

ax2.errorbar(data.l2[::-1],data.stokesUn[::-1], yerr=data.noise, fmt='^', ms=3, c='red', capthick=0, lw=0.1, label='U')
ax2.plot(xufilt,stokesU_filt, c='red', lw=1)

ax2.axhline(y=0, ls=':', c='lightgray')
ax2.set_ylabel(r"Intensity [$\mu$Jy/beam]", fontsize=12)
ax2.set_xlabel(r"$\lambda^2$ [m$^2$]", fontsize=12)
ax2.legend()

# ----------
# pol fraction

ax3 = fig.add_axes([0.1, 0.7, 0.5, 0.5], xlim=(np.min(data.l2),np.max(data.l2)))

p_frac = np.sqrt(data.stokesQn**2 + data.stokesUn**2)/data.stokesI
frac0 = np.mean(p_frac)
l2mid = 0.5*(np.max(data.l2) - np.min(data.l2)) + np.min(data.l2)

bnum = make_bins(data.l2[::-1], p_frac[::-1])
for i in range(int(np.max(bnum)+1)):
    pos = np.mean(data.l2[::-1][np.where(bnum==i)])
    val = np.mean(p_frac[::-1][np.where(bnum==i)])
    std = np.std(p_frac[::-1][np.where(bnum==i)])
    xlim1 = pos - np.min(data.l2[::-1][np.where(bnum==i)])
    xlim2 = np.max(data.l2[::-1][np.where(bnum==i)]) - pos
    ax3.errorbar(pos, val, yerr=std, xerr=[[xlim1], [xlim2]], fmt='.', capsize=3, c='lightgray')
    
ax3.errorbar(data.l2[::-1],p_frac[::-1], fmt='.', c='black', capthick=0)
ax3.plot(data.l2[::-1],np.ones(data.l2.shape[0])*frac0, c='darkgray', ls='dashed')
ax3.text(l2mid-0.01, 0.995*np.max(p_frac), r"Average: {:.1f}%".format(frac0*100), color='gray', fontsize=12)
ax3.set_ylabel(r"Polarisation fraction", fontsize=12)
ax3.set_xlabel(r"$\lambda^2$ [m$^2$]", fontsize=12)

# ----------
# FD spectrum

data.cfg.pol_frac = True
data.cfg.bkg_correction = True

data.read_data()           # read QU data

fspec = []
w = np.ones(len(data.stokesQn))
phi = np.linspace(-500,500,10000)

for i in range(0,len(phi)):
    fspec.append(calc_f(phi[i],data.l2[::-1],data.stokesQn[::-1],data.stokesUn[::-1],w))

phi0 = phi[np.argmax(np.abs(fspec))]

ax5 = fig.add_axes([0.1, -0.4, 1.1, 0.4], xlim=(np.min(phi),np.max(phi)))
ax5.text(320, 0.95*np.max(np.abs(fspec)), r"peak @ {:.2f} rad/m$^2$".format(phi0), color='gray', fontsize=12)
ax5.plot(phi, np.abs(fspec), c='black')
ax5.axvline(x=phi0, ls=':', c='lightgray')
ax5.set_ylabel(r"|P| [mJy beam$^{-1}$ rmtf$^{-1}$]", fontsize=12)
ax5.set_xlabel(r"$\phi$ [rad m$^{-2}$]", fontsize=12)

# ----------
# pol angle

ax4 = fig.add_axes([0.7, 0.7, 0.5, 0.5], xlim=(np.min(data.l2),np.max(data.l2)))

p_angl = np.arctan2(data.stokesUn,data.stokesQn)
pfit = fit_chi(data.l2[::-1],p_angl[::-1],data.noise,[phi0,0.])
#p_angl = np.unwrap(p_angl)  # <---- don't use this (because noise fluctuations)
p_angl*=0.5

l2mid = 0.5*(np.max(data.l2) - np.min(data.l2)) + np.min(data.l2)

ax4.errorbar(data.l2[::-1],p_angl[::-1], fmt='.', c='black', capthick=0)
ax4.plot(data.l2[::-1], line(0.5*pfit,data.l2[::-1]), c='lightgray')
ax4.plot(data.l2[::-1], line(0.5*pfit,data.l2[::-1])-np.pi, c='lightgray')
ax4.plot(data.l2[::-1], line(0.5*pfit,data.l2[::-1])+np.pi, c='lightgray')
ax4.text(l2mid-0.015, 1.4, r"d$\chi$/d$\lambda^2$ = {:.2f} rad/m$^2$".format(0.5*pfit[0]), color='gray', fontsize=12)
ax4.set_ylabel(r"Polarisation angle, $\chi$ [radians]", fontsize=12)
ax4.set_xlabel(r"$\lambda^2$ [m$^2$]", fontsize=12)
ax4.set_ylim(-0.5*np.pi, 0.5*np.pi)

# ----------

srcid = cfg.data_file.split('_')[0].strip('ID')
fig.suptitle("ID{}".format(srcid), x=0.6, y=1.25, fontsize=16)
fig.savefig("ID{}_polpanel.png".format(srcid), bbox_inches = 'tight')
