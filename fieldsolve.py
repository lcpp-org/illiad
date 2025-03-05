## IMPORTS
import pandas as pd
import numpy as np
from time import perf_counter

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
#from coordtrans import RTP_to_XYZ

#################
## USER INPUTS ##
#################

## NAME YOUR OUTPUT FILE
output_name = 'It486_Ih900_Iv000_0p943_1p00'
#output_name = 'FITTED_02092025_hel-0p950'

## DEFINE MESH RESOLUTION
test = [20, 4, 10]
rough  = [  96,  90,  90 ] # dr=0.002m., dtheta=4deg., dphi=4deg.
lo_res = [  96,  90, 180 ] # dr=0.002m., dtheta=4deg., dphi=2deg.
hi_res = [ 191, 180, 360 ] # dr=0.001m., dtheta=2deg., dphi=1deg.
mesh_size = hi_res

# REF. COIL CURRENTS NORMALLY RUN ON HIDRA
#############################
#      | [Amp]| [Amp]| [Amp]#
# IOTA |  I_T |  I_H |  I_V #
# 1/3  |  486 |  900 |   00 #
# 1/4  |  486 |  790 |   00 #
# 1/5  |  486 |  710 |   00 #
# 1/7  |  581 |  581 |   00 #
# MAX. | 3500 | 7000 |   ?? #
#############################
# INPUT COIL CURRENTS:
I_toro = 486.
I_heli = 900.
I_vert = 0.

# *TESTING* Multiplier applied to all currents
# Used to model the attenuation of the fields due to the stainless-steel vacuum vessel
att_mult = 0.943080960048148 #0.967 #0.7
I_toro *= att_mult
I_heli *= att_mult
I_vert *= att_mult

# Multiplier applied to helical current,
# Used in conjunction with Cartesian error field to reproduce HIDRA's actual B-field
# Based on characterization of WEGA by Otte[REF] (Set to 1.0 if ideal field is desired)
err_mult = 1.0 #0.955
I_heli *= err_mult

########################
## END OF USER INPUTS ##
########################

def biotsavart_mesh(mesh, filament, current, Npoints):
    """ Solves for the magnetic field on a mesh of cartesian values 
    by using the Biot-Savart Law along a single filament 
    of current, returns Cartesian Field Vectors"""

    Rv = torch.zeros(mesh.shape).to(device)
    B = torch.zeros(mesh.shape).to(device)

    for i in range(Npoints):
        P1 = filament[:,i-1]
        P2 = filament[:,i]

        dl = P2 - P1
        dI = current * dl

        midpoint = (P1 + P2) / 2
        Rv = (mesh.T - midpoint)

        Rm = torch.sqrt( Rv[:,:,:,0]*Rv[:,:,:,0] +Rv[:,:,:,1]*Rv[:,:,:,1] + Rv[:,:,:,2]*Rv[:,:,:,2] ).T
        Rm3 = Rm*Rm*Rm

        B += torch.clone( torch.linalg.cross( dI.expand_as(Rv), Rv, dim=3 ).T / Rm3[:None] )
        
    return 1.0e-7 * B # ( mu_0/(4*pi) = 1E-07 )

def loop_through_coils(Bxyz, xyz_mesh, mycoils, coiltype, turns):
    """ Loops through each coil in 'mycoils' and sums each coil's contribution to the 
     total field 'Bxyz' on the mesh of cartesian values: 'xyz_mesh """
    current = torch.float

    ## CALLS FIELDSOLVER FOR EVERY CURRENT LOOP, SUMS RESULTS
    tick  = perf_counter()
    for n, coil in enumerate(mycoils):

        print('Coil({:02d}'.format(n+1)+'/{:02d}) '.format(len(mycoils))+coiltype[n])
        if coiltype[n] == 'Helix':                 current = turns[n] * I_heli # * 0.955 # Otte's error field correction
        elif coiltype[n] == 'toroidal_field':      current = turns[n] * I_toro
        elif coiltype[n] == 'Vertical_Field_Coil': current = turns[n] * I_vert
        else: print('COIL-TYPE ERROR!')

        coilpts = np.asarray(coil, dtype=np.float64)
        thiscoil = torch.tensor(coilpts) #, dtype=torch.float64)
        filament = thiscoil.T[:3].to(device)

        ## Mesh-ified
        N = filament.shape[1]
        Bxyz += biotsavart_mesh(xyz_mesh, filament, current, N)


    tock = perf_counter()
    print('Solution took {:.5f}s'.format(tock-tick))

    return Bxyz

def main():
    ## READ COIL INPUT FILE
    coilfile = "input_files/coils.wega_with_VFCoils"
    coildata = pd.read_csv(
    coilfile,
    header=None,
    skiprows=3,
    index_col=None,
    #delim_whitespace=True,
    sep='\s+',
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

    Rmajor = 0.72 #[m]
    Rminor = 0.19 #[m]

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
    print(' min. r={}, min. theta={}, min. phi={}'.format(r_minimum, theta_minimum, phi_minimum))
    ## END MESH PERIODICITY SETUP

    ## CREATE REGULARLY-=SPACED ARRAYS FOR EACH COORDINATE
    i_R     = torch.linspace(     r_minimum,     r_maximum,     nr)#.astype(np.float64)
    i_THETA = torch.linspace( theta_minimum, theta_maximum, ntheta)#.astype(np.float64)
    i_PHI   = torch.linspace(   phi_minimum,   phi_maximum,   nphi)#.astype(np.float64)

    # Create Cartesian Mesh
    rr, tt, pp = torch.meshgrid(i_R, i_THETA, i_PHI)

    xx = (Rmajor + rr*torch.cos(tt))*torch.cos(pp)
    yy = -(Rmajor + rr*torch.cos(tt))*torch.sin(pp)
    zz = rr*torch.sin(tt)
    xyz_mesh = torch.stack([xx, yy, zz]).to(device)

    # Loop through coils, summing each one's contribution to get total field
    Bxyz = torch.zeros(( 3, nr, ntheta, nphi )).to(device)
    Bxyz = loop_through_coils(Bxyz, xyz_mesh, mycoils, coiltype, turns)


    np.save(output_name, Bxyz.cpu())

if __name__ == '__main__':
    main()