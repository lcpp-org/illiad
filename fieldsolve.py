 ############
 ## IMPORTS
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.special as special
from numba import jit, prange
import numba as nb

## READ COIL INPUT FILE
coilfile = "coils.wega_with_VFCoils"
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
Rmajor = np.float64
Rmajor = 0.72 #[m]
Rminor = 0.19 #[m]

## DEFINE MESH RESOLUTION
##mesh_spacings = dr[meters], dtheta[radians], dphi[radians]
rough  = [ 0.003, 4.*(np.pi/180), 4.*(np.pi/180.)]
lo_res = [ 0.002, 4.*(np.pi/180), 2.*(np.pi/180.)]
hi_res = [ 0.001, 2.*(np.pi/180), 1.*(np.pi/180.)]

## DEFINE MESH PERIODICITY
## 0: NOT PERIODIC
## 1: 2PI PERIODIC
## >1: HIGHER PERIODICITY (i.e (2PI)/N  PERIODIC)
mesh_periodicity = [ 0, 1, 5]

## SELECT RESOLUTION
dr, dtheta, dphi = hi_res


r_prd, theta_prd, phi_prd = mesh_periodicity

r_maximum     =  (Rminor) / max(1, r_prd)
theta_maximum = (2*np.pi) / max(1, theta_prd)
phi_maximum   = (2*np.pi) / max(1, phi_prd)


## IF THE DIMENSION IS NOT PERIODIC, START AT 0
## IF IT IS PERIODIC, START AT DX (WHERE X IS THE COORDINATE)
r_minimum     =     dr * min(1, r_prd)
theta_minimum = dtheta * min(1, theta_prd)
phi_minimum   =   dphi * min(1, phi_prd)

## GET PROPER GRID SPACING
## (DEPENDS ON IF WE ARE STARTING AT 0 OR DX (WHERE X IS THE COORDINATE)
if r_prd == 0:
    nr = int((r_maximum + dr)/dr)
else: 
    nr = int(r_maximum/dr)

if theta_prd == 0:
    ntheta = int((theta_maximum + dtheta)/dtheta)
else: 
    ntheta = int(theta_maximum/dtheta)

if phi_prd == 0:
    nphi = int((phi_maximum + dphi)/dphi)
else: 
    nphi = int(phi_maximum/dphi)

## /END stuff that should be a mesh class method


## /START maybe more stuff that should be a mesh class method
## CREATE REGULARLY-=SPACED ARRAYS FOR EACH COORDINATE
i_R     = np.linspace(     r_minimum,     r_maximum,     nr).astype(np.float64)
i_THETA = np.linspace( theta_minimum, theta_maximum, ntheta).astype(np.float64)
i_PHI   = np.linspace(   phi_minimum,   phi_maximum,   nphi).astype(np.float64)

print(f'R array length {i_R.size}: {i_R}')
print(f'THETA array length {i_THETA.size}: {np.degrees(i_THETA)}')
print(f'PHI array length {i_PHI.size}: {np.degrees(i_PHI)}')


@jit(nb.types.Array(nb.float64, 1, "C")
(nb.types.Array(nb.float64, 2, "C"), nb.types.Array(nb.float64, 2, "C"), nb.float64, nb.int8), 
nopython=True)
def biotsavart( filament, point, current, Npoints):
    ##  TAKES IN A SINGLE POINT,
    ## SOLVES B FIELD DUE TO EVERY POINT IN FILAMENT USING BIOT-SAVART LAW,
    ## RETURNS FOR CARTESIAN FIELD VECTORS 


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
            for k in prange(0, int(PHI.size)):
                point = np.zeros((3,1), dtype=np.float64)
                Bpoint = np.zeros(3, dtype=np.float64)
                point[0] = (Rmajor + R[i]*np.cos(THETA[j])) * np.cos(PHI[k]) #X
                point[1] = (Rmajor + R[i]*np.cos(THETA[j])) * np.sin(PHI[k]) #Y
                point[2] = -R[i]*np.sin(THETA[j]) #Z
    
                Bpoint = biotsavart(filament, point,  current, N)
                
                Bxcoil[i][j][k] = Bpoint[0]
                Bycoil[i][j][k] = Bpoint[1]
                Bzcoil[i][j][k] = Bpoint[2]
                
    
    return Bxcoil, Bycoil, Bzcoil


""" ### COIL CURRENTS
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
"""
Bxsum = np.zeros((i_R.size,i_THETA.size,i_PHI.size))
Bysum = np.zeros((i_R.size,i_THETA.size,i_PHI.size))
Bzsum = np.zeros((i_R.size,i_THETA.size,i_PHI.size))
#Bnorm = np.zeros((i_R.size,i_THETA.size,i_PHI.size))

## CALLS FIELDSOLVER FOR EVERY CURRENT LOOP, SUMS RESULTS
for n, coil in enumerate(mycoils):
    bbx = np.zeros((i_R.size, i_THETA.size, i_PHI.size), dtype= np.float64)
    bby = np.zeros((i_R.size, i_THETA.size, i_PHI.size), dtype= np.float64)
    bbz = np.zeros((i_R.size, i_THETA.size, i_PHI.size), dtype= np.float64)
    current = np.double
    
    print('Coil({:02d}'.format(n+1)+'/{:02d}) '.format(len(mycoils))+coiltype[n])
    if coiltype[n] == 'Helix':
        current = turns[n]*900
    elif coiltype[n] == 'toroidal_field':
        current = turns[n]*486
    elif coiltype[n] == 'Vertical_Field_Coil':
        current = turns[n]*0.
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
np.save('Bxyz_iota-1q3_hires_5Periodic', Bxyz)




#Bnorm = np.sqrt(Bxsum**2 + Bysum**2 + Bzsum**2)
#np.save('Bnorm_out', Bnorm)

"""
## PLOT POLOIDAL CROSS-SECTIONS
contours = np.linspace(0.010, 0.150, 30)
rr,tt = np.meshgrid(i_R,i_THETA)

for i, p in enumerate(i_PHI):
    Bnormplot = Bnorm[i][:][:]
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_rmax(Rminor)
    plt.contourf(np.transpose(tt),np.transpose(rr),Bnormplot.T, 30, cmap='cividis')
    plt.colorbar()
    plt.title(r'B-field magnitude of HIDRA, $\phi$={:2.0f}$\degree$'.format(p*180/np.pi))
    plt.savefig('plots\HIDRA-i4_phi={:2.0f}.png'.format(p*180/np.pi),dpi=300)
plt.show()
"""