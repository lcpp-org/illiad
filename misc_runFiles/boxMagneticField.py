import pandas as pd
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from AGmesh import *

df = pd.read_csv("input_files/large_box.csv")
print(df.head(10))

magMesh = Mesh()
magMesh.loadCartesianField("input_files/It3500_Ih6300_Iv000_1p000_1p000_64bit.npy", errField=True, att_mult="default_poloidal")
#print(magMesh.Bx.shape)


def returnMesh():
    hi_res = [ 191, 180, 360 ] # dr=0.001m., dtheta=2deg., dphi=1deg.
    mesh_size = hi_res
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
        
    i_R     = torch.linspace(     r_minimum,     r_maximum,     nr)#.astype(np.float64)
    i_THETA = torch.linspace( theta_minimum, theta_maximum, ntheta)#.astype(np.float64)
    i_PHI   = torch.linspace(   phi_minimum,   phi_maximum,   nphi)#.astype(np.float64)

    rr, tt, pp = torch.meshgrid(i_R, i_THETA, i_PHI)
    xx = (RMAJOR + rr*torch.cos(tt))*torch.cos(pp)
    yy = -(RMAJOR + rr*torch.cos(tt))*torch.sin(pp)
    zz = rr*torch.sin(tt)
    xyz_mesh = torch.stack([xx, yy, zz]).to(device)
    return xyz_mesh.cpu()

"""
xxs = returnMesh().numpy()[0]
yys = returnMesh().numpy()[1]
zzs = returnMesh().numpy()[2]
df2= pd.DataFrame(xxs)
print(df2)

Data frame is organized 3 X 191 X 180 X 72
3 - x y z
191 - for the radii (don't know if its the entire 0.38 or just half)
180 - for the theta (don't know if its the entire 360 with 2 degree steps or something else)
72 - for the phi
"""
rad18 = np.deg2rad(18)
transMatrix = np.array([[-np.cos(rad18), -np.sin(rad18), 0],[-np.sin(rad18), np.cos(rad18), 0], [0,0, -1]])
"""
To transform from X Y Z coordinate system set up in the code (+x at 18 degrees CW from the South Side and +y at 18 degrees CW from 
the East Side +Z towards the roof forming a right handed system)
to the XYZ coordinates according to tokamak energ (+x at North Side Split, +y at the East Side and +z towards the floor - also right handed)
"""
transMatrixInv = np.linalg.inv(transMatrix)
#print(transMatrix @ transMatrixInv)

B_list = []
for i in range(len(df["x"])):
    #print(i) if i<51 else ""
    x = df["x"][i]/1000
    y = df["y"][i]/1000
    z = df["z"][i]/1000    
    #print([x,y,z]) if i<51 else ""
    xyz_normalCoord = [x,y,z] @ transMatrixInv
    #print(xyz_normalCoord) if i<51 else ""
    Bxyz, phi = magMesh.interpField(xyz_normalCoord, Cart=True)
    Bxyz_newCoord = Bxyz
    #print(Bxyz)if i<51 else ""
    Bxyz_newCoord = Bxyz @ transMatrix
    #print(Bxyz_newCoord)if i<51 else ""
    
    mag = np.sqrt(sum(Bxyz_newCoord**2))
    Bxyz_newCoord = Bxyz_newCoord.tolist()
    Bxyz_newCoord.append(mag)
    B_list.append(Bxyz_newCoord)

df2 = pd.DataFrame(B_list, columns=["Bx", "By", "Bz", "B"])
print(df2.head(50))