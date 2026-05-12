"""
#------------------------------------------------------#
# GENERATING POINCARE PLOTS FOR HIDRA'S MAGNETIC FIELD #
#------------------------------------------------------#
#        COIL CURRENTS NORMALLY RUN ON HIDRA           #
#------------------------------------------------------#
#  IOTA  |   I_T   |   I_H   |   I_V   |  PHI FWD/REV  #
#        |  [Amp]  |  [Amp]  |  [Amp]  |     [deg]     #
#  1/3   |   486   |   900   |    00   |    324/???    #
#  1/4   |   486   |   790   |    00   |    180/144    #
#  1/5   |   486   |   710   |    00   |    360/???    #
#  1/7   |   581   |   581   |    00   |    ???/???    #
#  MAX.  |  3500   |  7000   |    ??   |    ???/???    #
#------------------------------------------------------#
"""
import numpy as np
import matplotlib.pyplot as plt
from classes.iohandler import IOHandler
from classes.meshNew import Mesh
from utility.coordtrans import RTP_XYZ_JAC, RTP_XYZ_JAC2
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEFINE FIELDS #
CURRENT_TOR = 0.486 #[kA]
CURRENT_HEL = 0.860 #[kA]
CONFIG_TOR = "default_toroidal"
CONFIG_HEL = "default_helical"

# DEFINE OUTPUT DIRECTORY #
OUTPUT_DIR = "Test_NewPoincare"

def main():
    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    simIO = IOHandler(OUTPUT_DIR) 
    simIO.startLog()

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.setErrorField()
    b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=True, att_mult=CONFIG_TOR)
    b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)

    # Create coordinate arrays
    rho = torch.linspace(b_hidra.r_min, b_hidra.r_max, b_hidra.nr, dtype=torch.float64, device=device)
    theta = torch.linspace(b_hidra.theta_min, b_hidra.theta_max, b_hidra.ntheta, dtype=torch.float64, device=device)
    phi_period = b_hidra.periodicity[2]
    phi = torch.linspace(
        b_hidra.phi_min, b_hidra.phi_max*phi_period, b_hidra.nphi*phi_period,
        dtype=torch.float64, device=device)
    
    # Create RTP points array
    RHO_grid, THETA_grid, PHI_grid = torch.meshgrid(rho, theta, phi, indexing='ij')
    rtp_points = torch.stack([RHO_grid, THETA_grid, PHI_grid], dim=-1)
    simIO.log.info(f"RTP points shape: {rtp_points.shape}")
    mesh_shape = rtp_points.shape[:-1]

    # Interpolate B field (cartesian components), transform to physical RTP components
    rtp_flat = rtp_points.reshape(-1, 3)
    B_xyz = b_hidra.interpField(rtp_flat, Cart=False) # return (Bx, By, Bz) shape (3, N)
    B_rtp_phys = RTP_XYZ_JAC2(rtp_flat, B_xyz, form='xyz2rtp')

    rho_grid = rtp_points[:,:,:,0]
    theta_grid = rtp_points[:,:,:,1]
    Rcyl = b_hidra.R0 + rho_grid * torch.cos(theta_grid)  # R0 + ρ cosθ

    # copy B_rtp_phys to B_rtp_contra
    B_rtp_contra = B_rtp_phys.clone()
    B_rtp_contra = B_rtp_contra.reshape(3, *mesh_shape) #.cpu().numpy()

    """## CONVERT TO CONTRAVARIANT COMPONENTS ON THE GRID
    # Singularity at rho=0 for Btheta_contravariant,
    # keep physical component there for alternate integration method
    B_rtp_contra[1,:,:,:] = B_rtp_contra[1,:,:,:] / torch.clamp(rho_grid[:,:,:], min=b_hidra.dr/1e12) #B^θ
    #B_rtp_contra[1,1:,:,:] = B_rtp_contra[1,1:,:,:] / rho_grid[1:,:,:] #B^θ
    B_rtp_contra[2,:,:,:] = B_rtp_contra[2,:,:,:] / Rcyl[:,:,:] #B^ϕ

    # # Average over the theta direction?
    #B_rtp_contra[1, 0, :, :] = torch.mean(B_rtp_contra[1, 1, :, :], dim=0, keepdim=True)
    #B_rtp_contra[:, 0, :, :] = torch.mean(B_rtp_contra[:, 0, :, :], dim=-2, keepdim=True)

    # COPY VALUES TO RHO=0
    #B_rtp_contra[:, 0, :, :] = B_rtp_contra[:, 1, :, :]"""

    ## RE-TRY ORIGINAL (CHATGPT) METHOD:
    # Br=0 at r=0 from symmetry
    # Clamp calculated values at 1e-6?
    # Btheta must be theta-independent
    # Bphi must be theta-independent
    B_rtp_contra[0,0,:,:] = 0.0 #B^ρ
    #B_rtp_contra[0,0,:,:] = torch.mean(B_rtp_contra[0, 0, :, :], dim=0, keepdim=True) #B^ρ

    B_rtp_contra[1,:,:,:] = B_rtp_contra[1,:,:,:] / torch.clamp(rho_grid[:,:,:], min=1e-28)
    Rcyl = b_hidra.R0 + torch.clamp(rho_grid[:,:,:], min=1e-28) * torch.cos(theta_grid)  # R0 + ρcosθ
    B_rtp_contra[2,:,:,:] = B_rtp_contra[2,:,:,:] / Rcyl[:,:,:] #B^ϕ

    B_rtp_contra[1, 0, :, :] = torch.mean(B_rtp_contra[1, 0, :, :], dim=0, keepdim=True)
    B_rtp_contra[2, 0, :, :] = torch.mean(B_rtp_contra[2, 0, :, :], dim=0, keepdim=True)


    # check for NaNa or Infs and reshape back to grid
    if torch.isnan(B_rtp_contra).any() or torch.isinf(B_rtp_contra).any():
        print("\tWARNING: NaN or Inf values found in B_rtp_contra!")
    np.save("Bfield_RTPcontra_Ih860_Reglrzd_clamp1e28-28.npy", B_rtp_contra.cpu().numpy())
    simIO.log.info(f'## SIM FINISHED! Saved contravariant B field components, shape: {B_rtp_contra.shape}\n\n')

if __name__ == "__main__":
    main()