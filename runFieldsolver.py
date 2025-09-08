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
output_name = 'It3500_Ih6300_Iv000_1p000_1p000_64bit'

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
I_toro = 3500.   #[Amps]
I_heli = 6300.   #[Amps]
I_vert = 0000.   #[Amps]

s_toro = 1.0
s_heli = 1.0
s_vert = 1.0

I_toro *= s_toro
I_heli *= s_heli
I_vert *= s_vert

########################
## END OF USER INPUTS ##
########################

def biotsavart_mesh(mesh, filament, current, Npoints):
    """Calculates the magnetic field on a mesh of Cartesian points using the Biot-Savart Law for a single current filament.
    Args:
        mesh (torch.Tensor): A tensor of shape (..., 3) representing the Cartesian coordinates of the mesh points where the magnetic field is to be calculated.
        filament (torch.Tensor): A tensor of shape (3, Npoints) representing the 3D coordinates of the filament points defining the current path.
        current (float or torch.Tensor): The current (in Amperes) flowing through the filament.
        Npoints (int): The number of points along the filament (must match filament's second dimension).
    Returns:
        torch.Tensor: A tensor of the same shape as `mesh`, representing the magnetic field vectors (in Tesla) at each mesh point.
    Notes:
        - The calculation uses the Biot-Savart Law and returns the field in SI units (Tesla).
        - The result is scaled by 1.0e-7 (i.e., μ₀/(4π)).
        - Assumes all tensors are on the same device and have compatible shapes.
    """

    B = torch.zeros(mesh.shape, dtype=torch.float64, device=device)

    for i in range(Npoints):
        P1 = filament[:,i-1] #[meters]
        P2 = filament[:,i] #[meters]

        dl = P2 - P1
        dI = current * dl # [Amps*meters]

        midpoint = (P1 + P2) / 2
        Rv = (mesh.T - midpoint)

        Rm = torch.norm(Rv, dim=3).T
        Rm3 = Rm*Rm*Rm

        B += torch.linalg.cross( dI.expand_as(Rv), Rv, dim=3 ).T / Rm3[:None]
        
    return 1.0e-7 * B #[Tesla]

def loop_through_coils(Bxyz, xyz_mesh, mycoils, coiltype, turns):
    """
    Computes and sums the magnetic field contributions from multiple coils over a mesh grid.
    For each coil in `mycoils`, determines the current based on its type and number of turns,
    computes the magnetic field using the Biot-Savart law, and accumulates the result in `Bxyz`.
    Args:
        Bxyz (np.ndarray or torch.Tensor): The array or tensor to accumulate the total magnetic field (shape: [..., 3]).
        xyz_mesh (np.ndarray or torch.Tensor): The mesh grid of Cartesian coordinates where the field is evaluated.
        mycoils (list): List of coil definitions, each as an array-like of points describing the coil geometry.
        coiltype (list of str): List of coil type strings (e.g., 'Helix', 'toroidal_field', 'Vertical_Field_Coil').
        turns (list or np.ndarray): Number of turns for each coil.
    Returns:
        np.ndarray or torch.Tensor: The updated magnetic field array/tensor `Bxyz` with all coil contributions summed.
    """
    ## CALLS FIELDSOLVER FOR EVERY CURRENT LOOP, SUMS RESULTS
    tic  = perf_counter()
    for n, coil in enumerate(mycoils):

        print('Coil({:02d}'.format(n+1)+'/{:02d}) '.format(len(mycoils))+coiltype[n])
        if coiltype[n] == 'Helix':                 current = turns[n] * I_heli # * 0.955 # Otte's error field correction
        elif coiltype[n] == 'toroidal_field':      current = turns[n] * I_toro
        elif coiltype[n] == 'Vertical_Field_Coil': current = turns[n] * I_vert
        else: 
            raise ValueError(f"COIL-TYPE ERROR! Unknown coil type: {coiltype[n]}")

        coilpts = np.asarray(coil, dtype=np.float64)
        thiscoil = torch.tensor(coilpts) #, dtype=torch.float64)
        filament = thiscoil.T[:3].to(device)
        thiscoil = torch.tensor(coilpts, dtype=torch.float64, device=device)
        ## Mesh-ified
        N = filament.shape[1]
        Bxyz += biotsavart_mesh(xyz_mesh, filament, current, N)


    toc = perf_counter()
    print('Solution took {:.5f}s'.format(toc-tic))

    #return Bxyz

def main():
    """Main function to set up the mesh, read coil data, and compute the magnetic field using Biot-Savart law.
    It initializes the mesh parameters, reads coil data from a file, and computes the magnetic field.
    The results are saved to a specified output file."""

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


    RMAJOR = 0.72 #[m]
    RMINOR = 0.19 #[m]

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
        r_maximum = RMINOR*r_prd
        dr = r_maximum/nr
        r_minimum = dr
    else:
        r_maximum = RMINOR
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
    xx = (RMAJOR + rr*torch.cos(tt))*torch.cos(pp)
    yy = -(RMAJOR + rr*torch.cos(tt))*torch.sin(pp)
    zz = rr*torch.sin(tt)
    xyz_mesh = torch.stack([xx, yy, zz]).to(device)

    # Loop through coils, summing each one's contribution to get total field
    B_XYZ = torch.zeros(( 3, nr, ntheta, nphi ), dtype=torch.float64, device=device) #.to(device)
    loop_through_coils(B_XYZ, xyz_mesh, mycoils, coiltype, turns)
    np.save(output_name, B_XYZ.cpu())

if __name__ == '__main__':
    main()