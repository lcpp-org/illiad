 ############
 ## IMPORTS
import pandas as pd
import numpy as np

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.special as special
#from numba import jit, prange
#import numba as nb

from utility.coordtrans import RTP_to_XYZ

Rmaj = 0.72 #[m]
Rmin = 0.19 #[m]

## READ COIL INPUT FILE
coilfile = "input_files/coils.wega_with_VFCoils"
coildata = pd.read_csv(
coilfile,
header=None,
skiprows=3,
index_col=None,
delim_whitespace=True,
names=range(6)) #irregularly-sized rows, pad with NaN's

## PARSE COIL DATA
coil_delim = coildata.loc[coildata[5].isnull()==False].index.values.tolist() # find index of rows with coil tag

coiltype = coildata.iloc[coil_delim][5].values
turns = [coildata.iloc[i-1][3] for i in coil_delim]

mycoils = [None]*len(coil_delim)
for i, dum in enumerate(coil_delim):
    if i==0:
        mycoils[i] = coildata.iloc[:coil_delim[i], 0:4]
    else:
        mycoils[i] = coildata.iloc[coil_delim[i-1]+1:coil_delim[i], 0:4]


## /START stuff that should be a mesh class method
## DEFINE GEOMETRY
output_name = 'Bxyz_iota-1q3_MAXPOWER_hires_95p5pct'

Rmajor = np.float64
Rmajor = 0.72 #[m]
Rminor = 0.19 #[m]


## DEFINE MESH RESOLUTION
rough  = [  96,  90,  90 ] # dr=0.002m., dtheta=4deg., dphi=4deg.
lo_res = [  96,  90, 180 ] # dr=0.002m., dtheta=4deg., dphi=2deg.
hi_res = [ 191, 180, 5 ] # dr=0.001m., dtheta=2deg., dphi=1deg.

mesh_size = hi_res

## DEFINE MESH PERIODICITY
## 0: NOT PERIODIC
## 1: 2PI PERIODIC
## >1: HIGHER PERIODICITY (i.e (2PI)/N  PERIODIC)
mesh_periodicity = [ 0, 1, 5]

nr     = int( mesh_size[0] / max(1, mesh_periodicity[0]) )
ntheta = int( mesh_size[1] / max(1, mesh_periodicity[1]) )
nphi   = int( mesh_size[2] / max(1, mesh_periodicity[2]) )

r_prd, theta_prd, phi_prd = mesh_periodicity

print(f' nr = {nr}')
print(f' ntheta = {ntheta}')
print(f' nphi = {nphi}')


## IF THE DIMENSION IS NOT PERIODIC, START AT 0
## IF IT IS PERIODIC, START AT DX (WHERE X IS THE COORDINATE)
if r_prd:
    r_maximum = Rminor.r_prd
    dr = r_maximum/nr
    r_minimum = dr
else:
    r_maximum = Rminor
    dr = r_maximum/(nr-1)
    r_minimum = 0.

if theta_prd:
    theta_maximum = (2*np.pi) / theta_prd
    dtheta = theta_maximum/ntheta
    theta_minimum = dtheta
else:
    theta_maximum = (2*np.pi)
    dtheta = theta_maximum/(ntheta-1)
    theta_minimum = 0.
    
if phi_prd:
    phi_maximum = (2*np.pi) / phi_prd
    dphi = phi_maximum/nphi
    phi_minimum = dphi
else:
    phi_maximum = (2*np.pi)
    dphi = phi_maximum/(nphi-1)
    phi_minimum = 0.


print(f' r max. = {r_maximum}')
print(f' theta max. = {theta_maximum}')
print(f' phi max. = {phi_maximum}')

print(f' r min. = {r_minimum}')
print(f' theta min. = {theta_minimum}')
print(f' phi min. = {phi_minimum}')


## /START maybe more stuff that should be a mesh class method
## CREATE REGULARLY-=SPACED ARRAYS FOR EACH COORDINATE
R     = np.linspace(     r_minimum,     r_maximum,     nr).astype(np.float64)
THETA = np.linspace( theta_minimum, theta_maximum, ntheta).astype(np.float64)
PHI   = np.linspace(   phi_minimum,   phi_maximum,   nphi).astype(np.float64)

print(f'nr={nr}, R array length {R.size}: {R}')
print(f'ntheta={ntheta}, THETA array length {THETA.size}: {np.degrees(THETA)}')
print(f'nphi={nphi}, PHI array length {PHI.size}: {np.degrees(PHI)}')
X = np.zeros((R.size, THETA.size, PHI.size))
Y = np.zeros((R.size, THETA.size, PHI.size))
Z = np.zeros((R.size, THETA.size, PHI.size))

for i in range(0, int(R.size)):
    for j in range(0, int(THETA.size)):
        for k in range(0, int(PHI.size)):
            X[i][j][k] = (Rmajor + R[i]*np.cos(THETA[j])) * np.cos(PHI[k]) #X
            Y[i][j][k] = -(Rmajor + R[i]*np.cos(THETA[j])) * np.sin(PHI[k]) #Y
            Z[i][j][k] = R[i]*np.sin(THETA[j]) #Z


## INITIALIZE FIGURE
fig = plt.figure()#figsize=(12.8, 9.6)) #default: 6.4 x 4.8
ax = fig.add_subplot(projection='3d')

## PLOT VACUUM VESSEL TORUS
vvres = 100
# theta: poloidal angle; phi: toroidal angle
ptheta = np.linspace(0, 2*np.pi, vvres)
pphi   = np.linspace(0, 2.*np.pi, vvres)
ptheta, pphi = np.meshgrid(ptheta, pphi)

px = (Rmaj + Rmin*np.cos(ptheta)) * np.cos(pphi)
py = (Rmaj + Rmin*np.cos(ptheta)) * np.sin(pphi)
pz = Rmin * np.sin(ptheta)
ax.set_zlim(-0.7,0.7)

ax.plot_surface(px, py, pz, rstride=3, cstride=3, color='grey', edgecolor='dimgrey', linewidth=0.1, alpha=0.3, shade=True) #'dimgrey'
ax.scatter(X, Y, Z, 'o', c='k', s=0.1)

plt.title('HIDRA Mesh')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')
ax.grid(False)
plt.margins(0.05)

#plt.savefig('HIDRA_mesh.png', bbox_inches='tight', dpi=600)
plt.show()