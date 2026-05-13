## IMPORT
import os
import sys
# Allow running from any subdirectory: resolve the project root relative to this file
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


#import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# import classes.class_outputHandler as out
from classes.iohandler import IOHandler
#from classes.mesh import *
from classes.mesh import Mesh
from utility.coordtrans import RTP_to_XYZ, XYZ_to_RTP
from utility.coordtrans import RTP_XYZ_JAC, RTP_XYZ_JAC2
'''
Things to change include
simIO out
input magnetic file to be loaded
angles for PHI
booleans for highToLow and deltas for getValuesAlong0
plot_XSection (comment out or not)

'''


## DEFINE FIELDS
FIELD_FILE_TOR = 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy'
FIELD_FILE_HEL = 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy'
CURRENT_TOR = 0.486 #[kA]
CURRENT_HEL = 0.790 #[kA]
CONFIG_TOR = 'default_toroidal'
CONFIG_HEL = 'default_helical'

OUTPUT_DIRECTORY_NAME = "Bfields_It-0486_Ih-0790"


## SET UP RUN DIRECTORY
simIO = IOHandler(OUTPUT_DIRECTORY_NAME) 
simIO.startLog()

## DEFINE MESH AND LOAD FIELD
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.setErrorField()
b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=True, att_mult=CONFIG_TOR)
b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)


mesh_nr = int(b_hidra.nr//2 + 1)
mesh_dr = b_hidra.r_min*2 # !! R_min=0?
mesh_ntheta = int(b_hidra.ntheta/2)
mesh_dtheta = b_hidra.dtheta*2
mesh_nphi = 40
mesh_dphi = 9*(np.pi/180)
R     = np.linspace( b_hidra.r_min*2, b_hidra.r_max, mesh_nr)
THETA = np.linspace( b_hidra.theta_min, b_hidra.theta_max, mesh_ntheta)
#PHI   = np.linspace( b_hidra.phi_min*2,     b_hidra.phi_max,   int(b_hidra.nphi/2))
PHI   = np.linspace( 9*(np.pi/180), 2*np.pi, mesh_nphi)
mesh_size = (R.size, THETA.size, PHI.size)

rr,tt = np.meshgrid(R,THETA)
rb,tb,pb = np.meshgrid(R,THETA,PHI)

# CALCULATE B-COMPONENTS #
Br = np.zeros(mesh_size)
Bpol = np.zeros(mesh_size)
Btor = np.zeros(mesh_size)
Bnorm = np.zeros(mesh_size)
Bvert = np.zeros(mesh_size)
iota_calc = np.zeros(mesh_size)
BR_cylnd = np.zeros(mesh_size)


for j, theta in enumerate(THETA):
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    for k, phi in enumerate(PHI):
        # replace with function call to coordtrans
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        # Xform = np.array([[ctheta*cphi, -ctheta*sphi, stheta],
        #                 [ -stheta*cphi,  stheta*sphi, ctheta],
        #                 [ -sphi, -cphi, 0]])


        for i, r in enumerate(R):
            bxyz, dum = b_hidra.interpField(np.asarray([r, theta, phi]), Cart=False)

            # replace with function call to coordtrans
            br, bpol, btor = RTP_XYZ_JAC(np.asarray([r, theta, phi]), bxyz, form='xyz2rtp')
            #br, bpol, btor = np.dot(Xform, bxyz)

            #if r == 0.:
            if i == 0:
                bpol = 0

            Bnorm[i][j][k] = np.sqrt(bxyz[0]**2 + bxyz[1]**2 + bxyz[2]**2)
            Br[i][j][k] = br
            Bpol[i][j][k] = bpol
            Btor[i][j][k] = btor
            Bvert[i][j][k] = bxyz[2]
            #iota_calc[i][j][k] = btor*r/(bpol*b_hidra.R0) if bpol != 0 else 0.
            #iota_calc[i][j][k] = btor*(b_hidra.R0 + r*ctheta)/(bpol*b_hidra.R0) if bpol != 0 else 0.

            iota_calc[i][j][k] = bpol/btor * r/(b_hidra.R0 + r*ctheta) if bpol != 0 else 0.

            #iota_calc[i][j][k] = 2 * np.pi * br * b_hidra.R0 / bpol / r if (bpol != 0 or r != 0) else 0.
            #iota_calc[i][j][k] = 2 * np.pi * btor * b_hidra.R0 / bpol / (b_hidra.R0 + r*ctheta) if (bpol != 0 or r != 0) else 0.
            #iota_calc[i][j][k] = 2 * np.pi * btor  / bpol if (bpol != 0 or r != 0) else 0.

            BR_cylnd[i][j][k] = bxyz[0] * cphi - bxyz[1] * sphi

print('Fields Calculated.')


def plot_Xsection(title, data, filename, phi_toPlot, contours=None):
    print('Plotting ' + title + '...')
    max_data = np.max(data)
    min_data = np.min(data)
    if contours is None:
        contours = np.linspace(min_data, max_data, 24)

    # Adding endpoint for continuous plot through origin
    wrped_tt = np.concatenate((tt, tt[-1:] + mesh_dtheta))#b_hidra.dtheta
    wrped_rr = np.concatenate((rr, rr[-1:]))

    for i, phi_comp in enumerate(phi_toPlot):
        plot_data = np.transpose(data, [2,1,0])[i]
        loc_max = np.max(plot_data)
        loc_min = np.min(plot_data)
    
        wrp_data = np.concatenate((plot_data, plot_data[0:1, :]), axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        plt.contourf(wrped_tt.T, wrped_rr.T, wrp_data.T, contours, cmap='viridis')

        ax.set_rmax(b_hidra.r_max)
        ax.set_rticks(np.arange(0.0, 0.19, 0.02))
        ax.yaxis.set_tick_params(labelsize=5)
        ax.grid(linewidth = 0.25, linestyle=':', c='k')

        plt.colorbar()

        phi_phys = (phi_comp + (198 * np.pi/180.)) % (2*np.pi)

        plt.title('$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split $(\phi_c$={:02.0f}$\degree)$\n'.format(phi_phys*180/np.pi, phi_comp*180/np.pi, loc_max)
                  + '$I_t$={:4.0f}A, $I_h$={:4.0f}A\n'.format(CURRENT_TOR*1000, CURRENT_HEL*1000)
                  + title + '\nloc. max = {:.4f}T\nloc. min = {:.4f}T'.format(loc_max, loc_min), loc='left', fontsize=8)

        plt.tight_layout()
        #plt.savefig(filename + '_phi={:02.0f}.png'.format(p*180/np.pi),dpi=300)
        plot_name = filename + '_phi={:02.0f}.png'.format(phi_comp*180/np.pi)
        simIO.saveFig(plot_name)
    plt.close()


# NORM ##
#plot_Xsection('B-field magnitude', Bnorm, 'Bnorm', PHI)

# RADIAL ##
#plot_Xsection('RADIAL B-field magnitude', Br, 'Bradial', PHI)
## POLOIDAL ##
#plot_Xsection('POLOIDAL B-field magnitude', Bpol, 'Bpoloidal', PHI)
## TOROIDAL ##
#plot_Xsection('TOROIDAL B-field magnitude', Btor, 'Btoroidal', PHI)
## VERTICAL ##
#plot_Xsection('VERTICAL B-field magnitude', Bvert, 'Bvertical', PHI)
## IOTA ##
plot_Xsection('IOTA', iota_calc, 'IOTA', PHI)#, contours=np.linspace(-1e0, 1e0, 24))
## Cylindrical Radial ##
plot_Xsection('Radial (Cyl.) B-field magnitude', BR_cylnd, 'BRadial_cyl', PHI)


"""
## WALL PLOTS
## Re-Do with [r][theta][phi] !
Bnormplot = np.transpose(Bnorm, [2,0,1])[-1]
print(Bnormplot.shape)

THETA = np.linspace( -np.pi, np.pi, ntheta)
PHI = np.linspace( 0, 2*np.pi, (nphi-1)*5)

Bnormplot= np.roll(Bnormplot,int(len(THETA)/2),axis=1)
Bnormplot2 = Bnormplot[:-1,:]
print(Bnormplot2.shape)

Bnormplot3 = np.tile(Bnormplot2.T, 5)
print(Bnormplot3.shape)

phi, theta = np.meshgrid(np.degrees(PHI),np.degrees(THETA))
#phi, theta = np.meshgrid(np.degrees(PHI),np.degrees(THETA)[-15:4])

fig = plt.figure()
ax = fig.add_subplot(111, polar=False)
plt.contourf(np.transpose(phi),np.transpose(theta),Bnormplot3.T, contours, cmap='viridis')
plt.colorbar()
#plt.contour(np.transpose(phi),np.transpose(theta),Bnormplot2.T, contours, linewidths=0.2, colors='k' )
plt.xlabel('Toroidal Angle $\phi [\degree]$')
plt.xticks(np.arange(0,360, step=36))
plt.ylabel('Poloidal Angle $\theta [\degree]$')
plt.yticks(np.arange(-180,180, step=45))
plt.title(r'B-field magnitude of HIDRA (on wall)')
plt.savefig('HIDRA-i3_r=wall.png',dpi=300)
#plt.show()



Brplot = np.transpose(Br, [2,0,1])[-1]
print(Brplot.shape)

THETA = np.linspace( -np.pi, np.pi, ntheta)
PHI = np.linspace( 0, 2*np.pi, (nphi-1)*5)

Brplot= np.roll(Brplot,int(len(THETA)/2),axis=1)
Brplot2 = Brplot[:-1,:]
print(Brplot2.shape)

Brplot3 = np.tile(Brplot2.T, 5)
print(Brplot3.shape)

phi, theta = np.meshgrid(np.degrees(PHI), np.degrees(THETA))
#phi, theta = np.meshgrid(np.degrees(PHI),np.degrees(THETA)[-15:4])

fig = plt.figure()
ax = fig.add_subplot(111, polar=False)
plt.contourf(np.transpose(phi),np.transpose(theta),Brplot3.T, contoursr, cmap='viridis')
plt.colorbar()
#plt.contour(np.transpose(phi),np.transpose(theta),Brplot2.T, contours, linewidths=0.2, colors='k' )
plt.xlabel('Toroidal Angle $\phi [\degree]$')
plt.xticks(np.arange(0,360, step=36))
plt.ylabel('Poloidal Angle $\theta [\degree]$')
plt.yticks(np.arange(-180,180, step=45))
plt.title(r'Radial B-field magnitude of HIDRA (on wall)')
plt.savefig('Bradial-i3_r=wall.png',dpi=300)
"""