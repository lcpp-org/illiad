## IMPORTS
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.special as special
from numba import jit, prange
import numba as nb
from coordtrans import RTP_to_XYZ

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
coiltype = coildata.iloc[coil_delim][5].values #parsing coil type 
turns = [coildata.iloc[i-1][3] for i in coil_delim] # parsing turns from row before delimiter

mycoils = [None]*len(coil_delim)
for i, dum in enumerate(coil_delim):
    if i==0:
        mycoils[i] = coildata.iloc[:coil_delim[i], 0:4]
    else:
        mycoils[i] = coildata.iloc[coil_delim[i-1]+1:coil_delim[i], 0:4]

## /START stuff that should be a mesh class method
## DEFINE GEOMETRY
output_name = 'Bxyz_iota-1q3_test_hires_95p5pct'

Rmajor = np.float64
Rmajor = 0.72 #[m]
Rminor = 0.19 #[m]

## DEFINE MESH RESOLUTION
rough  = [  96,  90,  90 ] # dr=0.002m., dtheta=4deg., dphi=4deg.
lo_res = [  96,  90, 180 ] # dr=0.002m., dtheta=4deg., dphi=2deg.
hi_res = [ 191, 180, 360 ] # dr=0.001m., dtheta=2deg., dphi=1deg.
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

print(f' nr = {nr}')
print(f' ntheta = {ntheta}')
print(f' nphi = {nphi}')
print(f' r max. = {r_maximum}')
print(f' theta max. = {theta_maximum}')
print(f' phi max. = {phi_maximum}')
print(f' r min. = {r_minimum}')
print(f' theta min. = {theta_minimum}')
print(f' phi min. = {phi_minimum}')

## /START maybe more stuff that should be a mesh class method
## CREATE REGULARLY-=SPACED ARRAYS FOR EACH COORDINATE
i_R     = np.linspace(     r_minimum,     r_maximum,     nr).astype(np.float64)
i_THETA = np.linspace( theta_minimum, theta_maximum, ntheta).astype(np.float64)
i_PHI   = np.linspace(   phi_minimum,   phi_maximum,   nphi).astype(np.float64)

print(f'nr={nr}, R array length {i_R.size}: {i_R}')
print(f'ntheta={ntheta}, THETA array length {i_THETA.size}: {np.degrees(i_THETA)}')
print(f'nphi={nphi}, PHI array length {i_PHI.size}: {np.degrees(i_PHI)}')

@jit(nb.types.Array(nb.float64, 1, "C")
(nb.types.Array(nb.float64, 2, "C"), nb.types.Array(nb.float64, 2, "C"), nb.float64, nb.int8), 
nopython=True)
def biotsavart( filament, point, current, Npoints):
    """ Solves for the magnetic field at a single point 
    by using the Biot-Savart Law along a single filament 
    of current, returns Cartesian Field Vectors"""
    
    B = np.zeros((3,1), dtype=np.float64)
    midpoint = np.zeros((Npoints, 3))
    i= np.int32

    for i in range(Npoints):
        P1 = filament[:,i-1]
        P2 = filament[:,i]
        dl = P2 - P1
        midpoint[i] = 0.5 * (P1 + P2)
        Rv = np.transpose(point) - midpoint[i]
        
        Rm = np.linalg.norm( Rv[0,:] )
        R3 = Rm**3 # + 1.0e-12
        dI = current * dl
        dB = 1.0e-7 * np.cross(dI,Rv) / R3 # mu_0/(4*pi) = 1E-07
        B[0,0] += dB[0,0]
        B[0,1] += dB[0,1]
        B[0,2] += dB[0,2]

    return B[0]

@jit(nopython=True, parallel=True, nogil=True)
def fieldsolver(R, THETA, PHI, filament, current, Rmajor):
    ## CONVERTS R-THETA-PHI POSITION TO X-Y-Z,
    ## CALLS BIOTSAVART FOR EVERY POINT IN THE R-PHI-THETA MESH, 
    ## OUTPUTS CARTESIAN FIELD VECTORS FOR EACH POINT IN MESH
    Bxcoil = np.zeros((R.size, THETA.size, PHI.size), dtype=np.float64)
    Bycoil = np.zeros((R.size, THETA.size, PHI.size), dtype=np.float64)
    Bzcoil = np.zeros((R.size, THETA.size, PHI.size), dtype=np.float64)

    N = int(filament[1].size) #len(filament[1])

    for i in prange(0, int(R.size)):
        for j in prange(0, int(THETA.size)):
            for k in range(0, int(PHI.size)):
                point = np.zeros((3,1), dtype=np.float64)
                Bpoint = np.zeros(3, dtype=np.float64)
                point[0] = (Rmajor + R[i]*np.cos(THETA[j])) * np.cos(PHI[k]) #X
                point[1] = -(Rmajor + R[i]*np.cos(THETA[j])) * np.sin(PHI[k]) #Y
                point[2] = R[i]*np.sin(THETA[j]) #Z
                #point = RTP_to_XYZ(np.array([R[i], THETA[j], PHI[k]]), Rmajor)

                Bpoint = biotsavart(filament, point,  current, N)

                Bxcoil[i][j][k] = Bpoint[0]
                Bycoil[i][j][k] = Bpoint[1]
                Bzcoil[i][j][k] = Bpoint[2]
                
    return Bxcoil, Bycoil, Bzcoil

### COIL CURRENTS
## HELICAL
# I_H = 900A (1/3)
# I_H = 790A (1/4)
# I_H = 710A (1/5)
# I_H = 581A (1/7)
## TOROIDAL
# I_T = 486A (1/3, 4, 5)
# I_T = 581A (1/7)
## VERTICAL
# I_V = 0

Bxsum = np.zeros((i_R.size,i_THETA.size,i_PHI.size))
Bysum = np.zeros((i_R.size,i_THETA.size,i_PHI.size))
Bzsum = np.zeros((i_R.size,i_THETA.size,i_PHI.size))

## CALLS FIELDSOLVER FOR EVERY CURRENT LOOP, SUMS RESULTS
for n, coil in enumerate(mycoils):
    current = np.double

    print('Coil({:02d}'.format(n+1)+'/{:02d}) '.format(len(mycoils))+coiltype[n])
    if coiltype[n] == 'Helix':
        current = turns[n] * 900 * 0.955 # Otte's error field correction
    elif coiltype[n] == 'toroidal_field':
        current = turns[n] * 486
    elif coiltype[n] == 'Vertical_Field_Coil':
        current = turns[n] * 0.
    else: print('COIL-TYPE ERROR!')

    thiscoil = np.asarray(coil, dtype=np.float64)
    filament = np.array(thiscoil.T[:3])

    bbx, bby, bbz = fieldsolver(i_R, i_THETA, i_PHI, filament, current, Rmajor)

    Bxsum += bbx
    Bysum += bby
    Bzsum += bbz
## /END maybe more stuff that should be a mesh class method


## OUTPUT ARRAY OF VECTORS AND ARRAY OF MAGNITUDE
Bxyz = np.array([Bxsum, Bysum, Bzsum])

np.save(output_name, Bxyz)

#Bnorm = np.sqrt(Bxsum**2 + Bysum**2 + Bzsum**2)
#np.save('Bnorm_iota-1q3_MAXPOWER_hires', Bnorm)

"""
## PLOT POLOIDAL CROSS-SECTIONS
contours = np.linspace(0.010, 0.150, 30)
rr,tt = np.meshgrid(i_R,i_THETA)

for i, p in enumerate(i_PHI):
    Bzplot = Bzsum[:][:][i]
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_rmax(Rminor)
    plt.contourf(np.transpose(tt),np.transpose(rr),Bzplot.T, 30, cmap='cividis')
    plt.colorbar()
    plt.title(r'Bz of HIDRA, $\phi$={:2.0f}$\degree$'.format(p*180/np.pi))
    plt.savefig('plots\HIDRA-i4_phi={:2.0f}.png'.format(p*180/np.pi),dpi=300)
plt.show()
"""
