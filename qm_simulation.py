from math import *
from numpy import *
import numpy as np
import pylab
import matplotlib.pyplot as plt
# Set pylab to interactive mode so plots update when run outside ipython
pylab.ion()

print('One-dimensional Schrodinger equation - time evolution')

def fillax(x,y,*args,**kw):
    """Fill the space between an array of y values and the x axis.
    All args/kwargs are passed to the pylab.fill function.
    Returns the value of the pylab.fill() call.
    """

    xx = np.concatenate((x,np.array([x[-1],x[0]],x.dtype)))
    yy = np.concatenate((y,np.zeros(2,y.dtype)))
    return pylab.fill(xx, yy, *args,**kw)

#Physical input parameters

N = 250           #Number of points in the space
hbar = 1.054e-34  #Planck constant[Js]
e_mass = 9.1e-31   #Electron mass[kg]
eVolt = 1.6e-19    #Electronvolt(1 eV)
conv_eV = 1/eVolt      #Conversion energy factor
del_x = 50e-9/N   # Size cell
dt = 0.8e-16      # Time step
T_tot = 30.0      # Times in ps

n_step = int(T_tot*1e-12/dt)
print('N steps :',n_step)

state = 1.0  # Eigenstate number of the first left potential well
 
ra = (0.5*hbar/e_mass)*(dt/del_x**2)       # Energy stability method parameter

print('Stability method condition (ra < 0.15): ra = ',ra)

DX = del_x*1e9

X = DX*np.linspace(0,N,N)     # Plot space (nm)
# ______________________________________________________________________
# Configuration of coupled wells

V = np.zeros(N)

V[99:101] = 0.2*eVolt    # First potential well height
V[149:151] = 0.2*eVolt   # Third potential well height

# Second potential well
V[102:148] = -0.003*eVolt

# Starting wave function

a = 100.0    # Width well
prl = np.zeros(N);
pim = np.zeros(N);
ptot = 0.0


for n in np.arange(2,100):
 prl[n] = np.sin(state*np.pi*n/a)
 ptot = ptot + prl[n]**2 + pim[n]**2

pnorm = sqrt(ptot)

# Normalizing and checking wave function

ptot = 0.;
for n in np.arange(0,N):    # v0123..
  prl[n] = prl[n]/pnorm
  pim[n] = pim[n]/pnorm
  ptot = ptot + prl[n]**2 + pim[n]**2

# ------------  FDTD loop------------
print("---- Calculing the FDTD loop ----")
for m in np.arange(1,n_step+1):
         for n in np.arange(1,N-2):
            prl[n] = prl[n] - ra*(pim[n-1] -2*pim[n] + pim[n+1]) + (dt/hbar)*V[n]*pim[n]
        
         for n in np.arange(1,N-2):
            pim[n] = pim[n] + ra*(prl[n-1] -2*prl[n] + prl[n+1]) - (dt/hbar)*V[n]*prl[n]

# Checking normalization

ptot = 0.0
for n in np.arange(0,N-1):
 ptot = ptot + prl[n]**2 + pim[n]**2

print('Probability (pnorm):  ',ptot)
       
#Calculate of the expected values

PE = 0.0
psi = np.zeros(N) + 1j*np.zeros(N)

for n in np.arange(0,N-1):
    psi[n] = prl[n] + 1j * pim[n]
    PE = PE + psi[n]*psi[n].conjugate()*V[n]
    
PE = PE*conv_eV
ke = 0.0 + 1j*0.0

for n in np.arange(1,N-2):
    lap_p = psi[n+1] - 2*psi[n] + psi[n-1]
    ke = ke + lap_p*psi[n].conjugate()
    
KE = -conv_eV*((hbar/del_x)**2/(2*e_mass))*ke.real

print('Electron energy (KE[eV]): ', KE)
# Plotting

pylab.figure()
pylab.plot(X,conv_eV*V,':k',zorder=0)   #  Potential line.
fillax(X,conv_eV*V, facecolor='y', alpha=0.2,zorder=0)

xmin = 0
xmax = X.max() 
Vmax = abs(V.max())*conv_eV
ymax = 1.8*Vmax
print('Vmax [eV] =',Vmax)

pylab.axis([xmin,xmax,-ymax,ymax])

print('abs  ',abs(prl.max()))

prl2 = prl**2
prl2 = np.sqrt(prl2)
pim2 = pim**2
pim2 = np.sqrt(pim2)

esc1 = ymax/(1.3*prl2.max())
esc2 = ymax/(1.3*pim2.max())
if esc1 > esc2:
     esc = esc2
else:
     esc = esc1

print(' esc  ',esc)

d_prob = prl**2 + pim**2
mKE = KE * 1000    #Energy Transformation in meV

lineR, = pylab.plot(X,prl*esc,'b',alpha=0.75,label='Real', linewidth=1.1)
lineI, = pylab.plot(X,pim*esc,'r',alpha=0.75,label='Imag', linewidth=1.1)
lineP, = pylab.plot(X,d_prob*esc,'k',alpha=0.75,label='Probability', linewidth=1.1)
pylab.axhline(KE,color='g',label='Energy',zorder=1)
pylab.xlabel('x(nm)')
pylab.ylabel(r'$\psi$')
pylab.title('Number State : %d' %(state), fontsize=7)
pylab.legend(loc='lower right')
pylab.text(35.0, 0.20, 'Time = %.1e ps' %(T_tot), fontsize=7)
pylab.text(35.0, 0.25, 'KE = %.1e meV' %(mKE), fontsize=7)

# Show and saving the figure.
pylab.ioff()
pylab.savefig('FDTD_result.png')
pylab.show()

