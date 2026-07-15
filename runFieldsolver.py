import torch
import argparse
import numpy as np
import pandas as pd
from time import perf_counter
from utility.run_config import load_inputs_json, merge_input_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_INPUTS = {
    "OUTPUT_NAME": "It0000_Ih1000_Iv000_1p000_1p000_64bit",
    "MESH_SIZE": [191, 180, 360],

    "I_TORO": 0.0,
    "I_HELI": 1000.0,
    "I_VERT": 0.0,

    "COILFILE": "input_files/coils.wega_with_VFCoils",
    "RMAJOR": 0.72,
    "RMINOR": 0.19,
    "MESH_PERIODICITY": [0, 1, 5],
}

_CLI_INPUTS = object()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ILLIAD Biot-Savart magnetic field generation.")
    parser.add_argument(
        "--inputs-json",
        default=None,
        help="Optional path to a JSON object overriding runFieldsolver.py defaults.",
    )
    return parser.parse_args()

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
        P1 = filament[:,i-1] # [meters]
        P2 = filament[:,i] # [meters]

        dl = P2 - P1
        dI = current * dl # [Amps*meters]

        midpoint = (P1 + P2) / 2
        Rv = (mesh.T - midpoint)

        Rm = torch.norm(Rv, dim=3).T
        Rm3 = Rm*Rm*Rm

        B += torch.linalg.cross( dI.expand_as(Rv), Rv, dim=3 ).T / Rm3[:None]
        
    return 1.0e-7 * B # [Tesla]

def loop_through_coils(Bxyz, xyz_mesh, mycoils, coiltype, turns, params):
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
        if coiltype[n] == 'Helix':                 current = turns[n] * params["I_HELI"] # * 0.955 # Otte's error field correction
        elif coiltype[n] == 'toroidal_field':      current = turns[n] * params["I_TORO"]
        elif coiltype[n] == 'Vertical_Field_Coil': current = turns[n] * params["I_VERT"]
        else: 
            raise ValueError(f"COIL-TYPE ERROR! Unknown coil type: {coiltype[n]}")

        coilpts = np.asarray(coil, dtype=np.float64)
        thiscoil = torch.tensor(coilpts) #, dtype=torch.float64)
        filament = thiscoil.T[:3].to(device)
        ## Mesh-ified
        N = filament.shape[1]
        Bxyz += biotsavart_mesh(xyz_mesh, filament, current, N)


    toc = perf_counter()
    print('Solution took {:.5f}s'.format(toc-tic))

    #return Bxyz

def main(input_overrides=_CLI_INPUTS):
    """Main function to set up the mesh, read coil data, and compute the magnetic field using Biot-Savart law.
    It initializes the mesh parameters, reads coil data from a file, and computes the magnetic field.
    The results are saved to a specified output file."""
    if input_overrides is _CLI_INPUTS:
        args = parse_args()
        input_overrides = load_inputs_json(args.inputs_json, "Fieldsolver inputs") if args.inputs_json else None
    params = merge_input_params(DEFAULT_INPUTS, input_overrides)

    ## READ COIL INPUT FILE
    coildata = pd.read_csv(
    params["COILFILE"],
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


    ## DEFINE MESH PERIODICITY
    ## 0: NOT PERIODIC
    ## 1: 2PI PERIODIC
    ## >1: HIGHER PERIODICITY (i.e (2PI)/N  PERIODIC)
    mesh_size = params["MESH_SIZE"]
    mesh_periodicity = params["MESH_PERIODICITY"]

    nr     = int( mesh_size[0] / max(1, mesh_periodicity[0]) )
    ntheta = int( mesh_size[1] / max(1, mesh_periodicity[1]) )
    nphi   = int( mesh_size[2] / max(1, mesh_periodicity[2]) )

    r_prd, theta_prd, phi_prd = mesh_periodicity

    ## IF THE DIMENSION IS NOT PERIODIC, START AT 0
    ## IF IT IS PERIODIC, START AT DX (WHERE X IS THE COORDINATE)
    if r_prd:
        r_maximum = params["RMINOR"]*r_prd
        dr = r_maximum/nr
        r_minimum = dr
    else:
        r_maximum = params["RMINOR"]
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
    xx = (params["RMAJOR"] + rr*torch.cos(tt))*torch.cos(pp)
    yy = -(params["RMAJOR"] + rr*torch.cos(tt))*torch.sin(pp)
    zz = rr*torch.sin(tt)
    xyz_mesh = torch.stack([xx, yy, zz]).to(device)

    # Loop through coils, summing each one's contribution to get total field
    B_XYZ = torch.zeros(( 3, nr, ntheta, nphi ), dtype=torch.float64, device=device) #.to(device)
    loop_through_coils(B_XYZ, xyz_mesh, mycoils, coiltype, turns, params)
    np.save(params["OUTPUT_NAME"], B_XYZ.cpu())

if __name__ == '__main__':
    main()
