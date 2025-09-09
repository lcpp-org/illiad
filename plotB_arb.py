## IMPORT
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# import classes.class_outputHandler as out
from classes.iohandler import IOHandler
#from classes.mesh import *
from classes.mesh import Mesh
from utility.coordtrans import RTP_to_XYZ, XYZ_to_RTP


# #FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
# FIELD_FILE_TOR = 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy'
# FIELD_SCALE_TOR = 0.9448
# INPUT_CURR_TOR = 3500. #486

# #FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
# FIELD_FILE_HEL = 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy'
# FIELD_SCALE_HEL = 0.955 * FIELD_SCALE_TOR
# INPUT_CURR_HEL = 0. #6300. #3150

# ERRFIELD_MAG = 1.5654e-4 #[Tesla]
# ERRFIELD_DIR_DEG = 271.5 #[degrees]

# TOROIDAL AND HELICAL MAGNETIC FIELDS
TOROIDAL_CURRENT = 3.50 #[kA]
HELICAL_CURRENT = 0.00 #[kA]
CONFIG_TOR = 'default_toroidal'
CONFIG_HEL = 'default_helical'


#DATA_FILE = 'input_files/new_pfc_loc_coords.csv'
DATA_FILE = 'input_files/large_box.csv'
#DATA_FILE = 'input_files/small_box.csv'

OUTPUT_DIRECTORY_NAME = "BFIELDS_090925"
OUTPUT_FILE_NAME = '3500-IT_0000-IH_new_pfc_loc'

## SET UP RUN DIRECTORY
simIO = IOHandler(OUTPUT_DIRECTORY_NAME) 
simIO.startLog()

## DEFINE MESH AND LOAD FIELD
# tor_mult_total = FIELD_SCALE_TOR * INPUT_CURR_TOR/1000.
# hel_mult_total = FIELD_SCALE_HEL * INPUT_CURR_HEL/1000.
# b_hidra = Mesh(R0=0.72, a=0.19)
# b_hidra.loadCartesianField(FIELD_FILE_TOR, att_mult=tor_mult_total, errField=True )
# b_hidra.addFieldPerturbation(FIELD_FILE_HEL, att_mult=hel_mult_total)
# b_hidra.set_nonPer_errField(ERRFIELD_MAG, ERRFIELD_DIR_DEG*np.pi/180.)

b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.setErrorField()
b_hidra.loadCartesianField(coilCurrent=TOROIDAL_CURRENT, errField=True, att_mult=CONFIG_TOR)
b_hidra.addFieldPerturbation(coilCurrent=HELICAL_CURRENT, att_mult=CONFIG_HEL)



## LOAD MESH OF DESIRED POINTS FROM CSV
## Data given in XYZ coords,
## +X pointing to North Split(phi_comp=+162deg), and +Z pointing *DOWN*!
points_CADxyz = simIO.loadCSV(DATA_FILE)
simIO.log.info(f'{points_CADxyz.shape=}')

## ROTATION TRANSFORM MATRIX:
# Rotates the basis 180 CCW about Y-, then -18 about Z-axis
phi_xform = -18 
cosphi_xform = np.cos(np.radians(phi_xform))
sinphi_xform = np.sin(np.radians(phi_xform))
xFormMatrix = np.array([[-cosphi_xform, -sinphi_xform, 0.0],
                       [-sinphi_xform,  cosphi_xform, 0.0],
                       [0.0,           0.0,          -1.0]])

#convert to RTP coordinates using coordtrans.
points_SIMxyz = np.zeros_like(points_CADxyz)
fields_SIMxyz = np.zeros_like(points_CADxyz)
fields_CADxyz = np.zeros_like(points_CADxyz)
for i, point_cad in enumerate(points_CADxyz):
    points_SIMxyz[i] = np.dot(xFormMatrix, point_cad/1000)  #convert from mm to m
    fields_SIMxyz[i] = b_hidra.interpField(points_SIMxyz[i], Cart=True)[0]
    fields_CADxyz[i] = np.dot(xFormMatrix, fields_SIMxyz[i])  # apply inverse rotation
#print(f'{points_SIMxyz[:, 0]=}')


#save output as a csv with header x,y,z,bx,by,bz anddata from points_CADxyz and fields_CADxyz
output_data = np.hstack((points_CADxyz, fields_CADxyz))
#np.savetxt(OUTPUT_FILE_NAME+'.csv', output_data, delimiter=',', header='x,y,z,bx,by,bz', comments='')
simIO.saveCSV(output_data, OUTPUT_FILE_NAME+'.csv', header='x,y,z,bx,by,bz')


# plot the points in 3D, with the markers colored by the B-field magnitude
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#sc = ax.scatter(*points_SIMxyz.T*1000, c=np.linalg.norm(fields_SIMxyz, axis=1), cmap='viridis', marker='o')
sc = ax.scatter(*points_CADxyz.T, c=np.linalg.norm(fields_CADxyz, axis=1), cmap='viridis', marker='o')
xs = np.linspace(-720, 720, 1000)
ax.plot(xs, np.sqrt(720**2 - xs**2))
ax.plot(xs, -np.sqrt(720**2 - xs**2))
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
plt.title('HIDRA B-field points in Cartesian coordinates (+X: North, +Y: East)')
plt.colorbar(sc, label='B-field magnitude (T)', shrink=0.8)
plt.tight_layout()
#elevation=2212150; azimuthal=221225; roll=0
ax.view_init(elev=-157, azim=-44, roll=0)

# Save the figure
plt.show()
#simIO.saveFig(OUTPUT_FILE_NAME+'.png', dpi=300)