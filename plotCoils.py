 ############
 ## IMPORTS
import pandas as pd
import numpy as np

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from coordtrans import XYZ_to_RTP
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hi_res = [ 191, 180, 360 ] # dr=0.001m., dtheta=2deg., dphi=1deg.
mesh_size = hi_res

def main():
    Rmajor = 0.72 #[m]
    Rminor = 0.19 #[m]

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

    ## DEFINE MESH PERIODICITY
    ## 0: NOT PERIODIC
    ## 1: 2PI PERIODIC
    ## >1: HIGHER PERIODICITY (i.e (2PI)/N  PERIODIC)
    #mesh_periodicity = [ 0, 1, 5]

    '''nr     = int( mesh_size[0] / max(1, mesh_periodicity[0]) )
    ntheta = int( mesh_size[1] / max(1, mesh_periodicity[1]) )
    nphi   = int( mesh_size[2] / max(1, mesh_periodicity[2]) )

    r_prd, theta_prd, phi_prd = mesh_periodicity

    ## IF THE DIMENSION IS NOT PERIODIC, START AT 0
    ## IF IT IS PERIODIC, START AT DX (WHERE X IS THE COORDINATE)
    if r_prd:
        r_maximum = Rminor*r_prd
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

    print(' nr={}, ntheta={}, nphi={}'.format(nr, ntheta, nphi))
    print(' max. r={}, max. theta={}, max. phi={}'.format(r_maximum, theta_maximum, phi_maximum))
    print(' min. r={}, min. theta={}, min. phi={}'.format(r_minimum, theta_minimum, phi_minimum))'''
    ## END MESH PERIODICITY SETUP

    ## CREATE REGULARLY-=SPACED ARRAYS FOR EACH COORDINATE
    #i_R     = torch.linspace(     r_minimum,     r_maximum,     nr)#.astype(np.float64)
    #i_THETA = torch.linspace( theta_minimum, theta_maximum, ntheta)#.astype(np.float64)
    #i_PHI   = torch.linspace(   phi_minimum,   phi_maximum,   nphi)#.astype(np.float64)

    # Create Cartesian Mesh
    #rr, tt, pp = torch.meshgrid(i_R, i_THETA, i_PHI)

    #xx = (Rmajor + rr*torch.cos(tt))*torch.cos(pp)
    #yy = -(Rmajor + rr*torch.cos(tt))*torch.sin(pp)
    #zz = rr*torch.sin(tt)
    #xyz_mesh = torch.stack([xx, yy, zz]).to(device)

    fig = plt.figure()#figsize=(12.8, 9.6)) #default: 6.4 x 4.8
    ax = fig.add_subplot()#projection='3d')
    '''## PLOT VACUUM VESSEL TORUS
    vvres = 100
    # theta: poloidal angle; phi: toroidal angle
    ptheta = np.linspace(0, 2*np.pi, vvres)
    pphi   = np.linspace(0, 2.*np.pi, vvres)
    ptheta, pphi = np.meshgrid(ptheta, pphi)

    px = (Rmajor + Rminor*np.cos(ptheta)) * np.cos(pphi)
    py = (Rmajor + Rminor*np.cos(ptheta)) * np.sin(pphi)
    pz = Rminor * np.sin(ptheta)
    ax.set_zlim(-0.7,0.7)

    ax.plot_surface(px, py, pz, rstride=3, cstride=3, color='grey', edgecolor='dimgrey', linewidth=0.1, alpha=0.3, shade=True) #'dimgrey'
    '''
    for n, coil in enumerate(mycoils):

        coilpts = np.asarray(coil, dtype=np.float64)
        thiscoil = torch.tensor(coilpts, dtype=torch.float64)
        filament = np.ascontiguousarray(thiscoil.T[:3])
        
        r = np.zeros(filament.shape[1])
        t = np.zeros(filament.shape[1])
        p = np.zeros(filament.shape[1])

        for i,point in enumerate(filament.T):
            rtp = XYZ_to_RTP(np.ascontiguousarray(point), Rmajor)
            r[i] = rtp[0]
            t[i] = np.degrees(rtp[1])
            p[i] = np.degrees(rtp[2])
            #X[i] = (r*np.cos(theta)) * np.cos(phi)
            #Y[i] = (-r*np.cos(theta)) * np.sin(phi)
            #Z[i] = r * np.sin(theta)
        
        if coiltype[n] == 'Helix':
            if float(coilpts[0][3]) < 0:    c = 'r'
            elif float(coilpts[0][3]) > 0:  c = 'g'
            """for num, th in enumerate(t):
                if p[num] == 0 or 360 or th == 0 or 360:
                    p[num] = np.nan
                    t[num] = np.nan"""
            ax.plot(p[45:-1], t[45:-1], c)
        elif coiltype[n] == 'toroidal_field':     ax.plot(p, t, c='b')
        #elif coiltype[n] == 'Vertical_Field_Coil': ax.plot(p, t, c='orange')
        #else: print('COIL-TYPE ERROR!')
        ax.vlines(180, 0, 360, 'black', linewidth=3)
        

    plt.title('HIDRA Mesh')
    plt.xlabel('Phi Angle')
    plt.ylabel('Theta Angle')

    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])
    #plt.axis('off')
    ax.grid(False)
    plt.margins(0.05)

    #plt.savefig('HIDRA_mesh.png', bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()