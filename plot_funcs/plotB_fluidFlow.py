## IMPORT
import os
import sys
# Allow running from any subdirectory: resolve the project root relative to this file
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# import classes.class_outputHandler as out
from classes.iohandler import IOHandler
#from classes.mesh import *
from classes.mesh import Mesh
from utility.coordtrans import RTP_to_XYZ, XYZ_to_RTP
from misc_runFiles.magnetic_fiel_function_fitter import Magnetic_function_fitter
from sklearn.metrics import r2_score


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
TOROIDAL_CURRENT = 0.0 #[kA]
HELICAL_CURRENT = 0.0 #[kA]
CONFIG_TOR = 'default_toroidal'
CONFIG_HEL = 'default_helical'

DATA_FILE = 'input_files/cad_corners.csv'
#DATA_FILE = 'input_files/new_pfc_loc_coords.csv'
#DATA_FILE = 'input_files/large_box.csv'
#DATA_FILE = 'input_files/small_box.csv'

OUTPUT_DIRECTORY_NAME = "BFIELDS_100725"
OUTPUT_FILE_NAME = '3500-IT_6300-IH_corners_expanded_box'

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

# Rotates the basis 27 degrees CW around the Z axis
phi_plate = 27
cosphi_plate = np.cos(np.radians(phi_plate))
sinphi_plate = np.sin(np.radians(phi_plate))
plateMatrix = np.array([[cosphi_plate,  sinphi_plate,  0.0],
                        [-sinphi_plate,  cosphi_plate,   0.0],
                        [0.0,                    0.0,   1.0]])
plateMatrixInv = np.linalg.inv(plateMatrix)

#inclination of the plate, high at the high field side and low at the low field side
theta_plate=5
costheta_plate = np.cos(np.radians(theta_plate))
sintheta_plate = np.sin(np.radians(theta_plate))
thetaMatrix = np.array([[costheta_plate, 0, sintheta_plate],
                        [0, 1, 0],
                        [-sintheta_plate, 0, costheta_plate]])
thetaMatrixInv = np.linalg.inv(thetaMatrix)


points_SIMxyz = np.zeros_like(points_CADxyz)
points_PLATExyz = np.zeros_like(points_CADxyz)
points_FLUIDxyz = np.zeros_like(points_CADxyz)
for i, point_cad in enumerate(points_CADxyz):
    points_PLATExyz[i] = np.dot(plateMatrix, point_cad/1000) #convert from mm to m
    points_FLUIDxyz[i] = np.dot(thetaMatrix, points_PLATExyz[i])

numPoints = 150
xmax, ymax, zmax = points_FLUIDxyz[0]
xmin, ymin, zmin = points_FLUIDxyz[-1]
xs = np.linspace(xmin, xmax, numPoints)
ys = np.linspace(ymin, ymax, numPoints)
zs = np.array([zmax, zmax-0.0015, zmax-0.003])#substrate level, 1.5 mm above and 3 mm above

interp_points = []
for x in xs:
    for y in ys:
        for z in zs:
            interp_points.append([x,y,z])
points_FLUIDxyz = np.array(interp_points)

points_PLATExyz = np.zeros_like(points_FLUIDxyz)
points_CADxyz = np.zeros_like(points_FLUIDxyz)
fields_SIMxyz = np.zeros_like(points_FLUIDxyz)
fields_CADxyz = np.zeros_like(points_FLUIDxyz)
fields_PLATExyz = np.zeros_like(points_FLUIDxyz)
fields_FLUIDxyz = np.zeros_like(points_FLUIDxyz)

for i, point in enumerate(points_FLUIDxyz):
    points_PLATExyz[i] = np.dot(thetaMatrixInv, point)
    points_CADxyz[i] = np.dot(plateMatrixInv, points_PLATExyz[i])
    point_SIM = np.dot(xFormMatrix, points_CADxyz[i])
    fields_SIMxyz[i] = b_hidra.interpField(point_SIM, Cart=True)[0]
    fields_CADxyz[i] = np.dot(xFormMatrix, fields_SIMxyz[i])
    fields_PLATExyz[i] = np.dot(plateMatrix, fields_CADxyz[i])
    fields_FLUIDxyz[i] = np.dot(thetaMatrix, fields_PLATExyz[i])

#save output as a csv with header x,y,z,bx,by,bz anddata from points_CADxyz and fields_CADxyz
output_data = np.hstack((points_CADxyz, fields_FLUIDxyz))
#np.savetxt(OUTPUT_FILE_NAME+'.csv', output_data, delimiter=',', header='x,y,z,bx,by,bz', comments='')
simIO.saveCSV(output_data, OUTPUT_FILE_NAME+'.csv', header='x,y,z,bx,by,bz')

def fit():
    test = Magnetic_function_fitter(f"output/{OUTPUT_DIRECTORY_NAME}/data/{OUTPUT_FILE_NAME}.csv")
    fields = ['bx', 'by', 'bz']
    BFit = []
    
    # old fit coefficients
    #df35_00_CAD = {"bx": [-6.48e-01, 1.46e-03, -9.72e-04, 4.29e-04, -7.16e-07, 5.96e-08, -4.90e-07, 8.67e-07, -3.08e-07, 1.94e-09], "by": [1.16e+00, -7.81e-04, -1.48e-03, -7.69e-04, -9.63e-08, 1.44e-06, 8.87e-07, -3.18e-08, 5.27e-07, 1.47e-08]}
    #df35_63_CAD = {"bx": [-1.89e-01, 9.72e-04, -1.74e-03, -1.75e-04, -5.01e-07, -2.62e-08, 1.25e-06, 1.19e-06, -1.73e-07, -1.11e-06], "by": [1.30e+00, -1.16e-03, -4.23e-04, -1.43e-04, -3.17e-07, 1.73e-06, 2.58e-06, -2.23e-06, -2.59e-06, -8.32e-07], "bz": [-8.90e-01, -1.10e-04, 4.16e-03, 1.44e-03, 7.62e-07, -9.81e-07, -1.56e-06, -4.55e-06, 1.60e-06, -1.90e-06]}
    #df_35_63_FLUID_new = {"bx":[8.83e-01, -1.96e+00, -6.31e-01, 9.34e-01, 9.68e-01, 1.93e+00, 1.73e+00, -2.11e+00, -4.59e+00, 1.04e+00], "by": [6.93e-01, -1.43e+00, 2.89e+00, 3.80e+00, 8.76e-01, -8.04e-01, -4.18e+00, -3.99e+00, -4.11e+00, 3.09e+00], "bz": [-5.89e-01, -6.29e-01, 3.75e+00, -1.09e+00, 1.66e+00, -2.89e+00, -1.22e+00, -2.78e+00, 6.89e+00, 1.19e+00]}

    fitCoeffs = []
    for i in fields:
        #print(i)
        deg = 2
        cart_cord, X_poly, fit_coeffs = test.fitter(deg, i)
        #test.fit_tester(fit_bx, i)
        #BFit.append(fit_b)
        #fit_coeffs = np.array(df35_63_CAD[i]) # to demonstrate old fits
        #fit_coeffs[1:4] = fit_coeffs[1:4]*1000
        #fit_coeffs[4:] = fit_coeffs[4:]*1000*1000
        BFit.append(np.dot(X_poly, fit_coeffs))
        print(f"Equation for {i}: {fit_coeffs[0]:0.2e} + {fit_coeffs[1]:0.2e}(x+x0) + {fit_coeffs[2]:0.2e}(y+y0) + {fit_coeffs[3]:0.2e}(z+z0) + {fit_coeffs[4]:0.2e}(x+x0)^2 + {fit_coeffs[5]:0.2e}(x+x0)(y+y0) + {fit_coeffs[6]:0.2e}(x+x0)(z+z0) + {fit_coeffs[7]:0.2e}(y+y0)^2 + {fit_coeffs[8]:0.2e}(y+y0)(z+z0) + {fit_coeffs[9]:0.2e}(z+z0)^2")
        print(f"The R2 score is {r2_score(np.dot(X_poly, fit_coeffs), test.magnetic_data[i]):0.5f}")
        fitCoeffs.append(np.array(fit_coeffs))

    BFit = np.array(BFit)
    return BFit.T, np.array(fitCoeffs).T

BFit, fit_coeffs = fit()

def fitVectors():
    magnitudes = []
    for i in BFit:
        #print(i)
        Bx, By, Bz = i
        #magnitudes.append(np.sqrt(Bx**2 + By**2 + Bz**2))
        magnitudes.append(By)
    print(min(magnitudes), max(magnitudes))
    
    # plot the points in 3D, with the vectors pointing in the direction of the B-field
    zpoints = len(zs)
    for j in range(zpoints):
        slice_FLUIDxyz = points_FLUIDxyz[j::zpoints]
        col = 1
        fields_FLUID = BFit[j::zpoints][:, col]
        #print(fields_FLUID[:][1])
        fig = plt.figure()
        plt.rc('axes', titlesize=13, labelsize=12)
        
        ax = fig.add_subplot(111)
        
        sc = ax.scatter(slice_FLUIDxyz.T[0, :]*1000, slice_FLUIDxyz.T[1, :]*1000, c=fields_FLUID, cmap='viridis', marker='s')
        
        ax.invert_yaxis()
        cbar = plt.colorbar(sc)
        cbar.set_label("B$_T$ (T)")
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.title(f'HIDRA B$_T$ fit in FLUID coordinates (+X: width of plate, +Y: length of plate) [{min(fields_FLUID):0.2e}T, {max(fields_FLUID):0.2e}T]')
        plt.tight_layout()
        plt.show()

#fitVectors()

radii = np.array([654.5, 662.5, 670.5, 678.5, 686.5, 694.5, 702.4, 710.4, 718.4, 726.4, 734.4, 742.4])
theta = np.deg2rad(27)
poloidal = np.deg2rad(np.array([8.78, 8.73, 8.69, 8.64, 8.60, 8.56, 8.52, 8.48, 8.44, 8.4, 8.37, 8.33]))

plate_points = []
field_Fluid_exact = []
field_Fluid_equation = []
for i, r in enumerate(radii):
    z = r*np.sin(poloidal[i])
    xy = r*np.cos(poloidal[i])
    x = xy*np.cos(theta)
    y = xy*np.sin(theta)
    #Trying to get everything into CAD coordinates from whatever it was before
    mmtom = 10**(-3)
    point_cad = np.array([x, y, z])/1000
    expanded = np.array([1, x*mmtom, y*mmtom, z*mmtom, x**2*mmtom*mmtom, x*y*mmtom*mmtom, x*z*mmtom*mmtom, y**2*mmtom*mmtom, y*z*mmtom*mmtom, z**2*mmtom*mmtom])
    point_SIM = np.dot(xFormMatrix, point_cad)
    field_SIM = b_hidra.interpField(point_SIM, Cart=True)[0]
    field_CAD = np.dot(xFormMatrix, field_SIM)
    field_PLATE = np.dot(plateMatrix, field_CAD)
    field_FLUID = np.dot(thetaMatrix, field_PLATE)
    field_Fluid_exact.append(field_FLUID)
    field_Fluid_equation.append(np.linalg.matmul(expanded, fit_coeffs))
    plate_points.append(point_cad)

field_Fluid_exact = np.array(field_Fluid_exact).T
field_Fluid_equation = np.array(field_Fluid_equation).T
#print(np.array(plate_points))
df = pd.DataFrame({"Interpx": field_Fluid_exact[0], "Interpy": field_Fluid_exact[1], "Interpz": field_Fluid_exact[2],"Equationx": field_Fluid_equation[0], "Equationy": field_Fluid_equation[1], "Equationz": field_Fluid_equation[2]})
df2 = pd.DataFrame({"fit_coeffs bx":fit_coeffs.T[0],"fit_coeffs by":fit_coeffs.T[1],"fit_coeffs bz":fit_coeffs.T[2]})

with pd.ExcelWriter('output.xlsx',  if_sheet_exists="overlay", mode="a") as writer:
    df.to_excel(writer, sheet_name=f'It_{TOROIDAL_CURRENT},Ih_{HELICAL_CURRENT}', index=False)
    df2.to_excel(writer, sheet_name=f"It_{TOROIDAL_CURRENT},Ih_{HELICAL_CURRENT}", startrow=16, index=False)


def points(): 
    # plot the points in 3D, with the markers colored by the B-field magnitude
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #sc = ax.scatter(*points_SIMxyz.T*1000, c=np.linalg.norm(fields_SIMxyz, axis=1), cmap='viridis', marker='o')
    sc = ax.scatter(*points_CADxyz.T, c=np.linalg.norm(fields_FLUIDxyz, axis=1), cmap='viridis', marker='o')
    #Overall B Strength
    #sc = ax.scatter(*points_CADxyz.T, c=np.abs((np.linalg.norm(fields_CADxyz, axis=1)-np.linalg.norm(BFit.T, axis=1))/np.linalg.norm(fields_CADxyz, axis=1)), cmap='viridis', marker='o')
    
    #sc = ax.scatter(*points_CADxyz.T, c=np.abs((fields_CADxyz[:, 2]-BFit[2])/np.linalg.norm(fields_CADxyz, axis=1)), cmap='viridis', marker='o')
    xs = np.linspace(-720, 720, 1000)
    ax.plot(xs, np.sqrt(720**2 - xs**2))
    ax.plot(xs, -np.sqrt(720**2 - xs**2))
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.title('HIDRA B-field points in Cartesian coordinates (+X: North, +Y: East)')
    plt.colorbar(sc, label='$\Delta B_z$ between Fit and Code Value (%)', shrink=0.8)
    plt.tight_layout()
    #elevation=2212150; azimuthal=221225; roll=0
    ax.view_init(elev=-157, azim=-44, roll=0)

    # Save the figure
    plt.show()
    #simIO.saveFig(OUTPUT_FILE_NAME+'.png', dpi=300)


#points()
def vectors():
    magnitudes = []
    for i in fields_FLUIDxyz:
        #print(i)
        Bx, By, Bz = i
        #magnitudes.append(np.sqrt(Bx**2 + By**2 + Bz**2))
        magnitudes.append(By)
    #print(magnitudes)
    
    # plot the points in 3D, with the vectors pointing in the direction of the B-field
    zpoints = len(zs)
    for j in range(zpoints):
        slice_FLUIDxyz = points_FLUIDxyz[j::zpoints]
        fields_FLUID = fields_FLUIDxyz[j::zpoints]
        #print(fields_FLUID[:][1])
        fig = plt.figure()
        plt.rc('axes', titlesize=13, labelsize=12)
        
        ax = fig.add_subplot(111)
        
        sc = ax.scatter(slice_FLUIDxyz.T[0, :]*1000, slice_FLUIDxyz.T[1, :]*1000, c=fields_FLUID[:, 1], cmap='viridis', marker='s')
        '''
        x_0,y_0,z_0 = point
        Bx, By, Bz = fields_FLUIDxyz[i]
        Bstrength = 0.02*np.sqrt(Bx**2 + By**2 + Bz**2)
        sc = ax.scatter(x_0, y_0, ,marker='o', color=colors_FLUID[i])
        '''
        ax.invert_yaxis()
        cbar = plt.colorbar(sc)
        cbar.set_label("B$_T$ (T)")
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        #ax.set_zlabel('Z (mm)')
        plt.title(f'HIDRA B$_T$ in FLUID coordinates (+X: width of plate, +Y: length of plate)')
        plt.tight_layout()
        #ax.view_init(elev=-34, azim=-125, roll=179)
        plt.show()


    
    """fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, point in enumerate(points_PLATExyz):
        x_0,y_0,z_0 = point
        Bx, By, Bz = fields_PLATExyz[i]
        Bstrength = 0.04*np.sqrt(Bx**2 + By**2 + Bz**2)
        ax.quiver(x_0, y_0, z_0, (Bx/Bstrength), (By/Bstrength), (Bz/Bstrength), colors=colors[i], arrow_length_ratio=0.05)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.title('HIDRA B-field points in PLATE coordinates (+X: width of plate, +Y: lenght of plate)')
    plt.tight_layout()
    ax.view_init(elev=-13, azim=-166, roll=180)
    plt.show()
    """
    '''
    xs = np.linspace(-720, 720, 1000)
    ax.plot(xs, np.sqrt(720**2 - xs**2))
    ax.plot(xs, -np.sqrt(720**2 - xs**2))
    '''
    
    #simIO.saveFig(OUTPUT_FILE_NAME+'.png', dpi=300)
#vectors()


"""
df = pd.read_csv(f"output/{OUTPUT_DIRECTORY_NAME}/data/{OUTPUT_FILE_NAME}.csv")










#BFit = (np.array(BFit).T).tolist()

# plot the points in 3D, with the vectors pointing in the direction of the B-field
fig, ax = plt.subplots()
x=9000
f = 9500
dif = 1
for i, point in enumerate(points_CADxyz[x:f:dif]):
    x_0,y_0,z_0 = point
    Bx, By, Bz = fields_CADxyz[i]
    Bstrength = np.sqrt(Bx**2 + By**2 + Bz**2)
    xaxis = 2
    ax.scatter(point[2], Bz, color="tab:blue")
    #ax.scatter(xaxis, By, color="tab:orange")
    #ax.scatter(xaxis, Bx, color="tab:green")
    #ax.scatter(xaxis, Bstrength, color="tab:red")

ax.plot(points_CADxyz[x:f:dif, xaxis], BFit[2][x:f:dif], color="tab:blue", label = "Bz")
#ax.plot(points_CADxyz[x:f:dif, 2], BFit[1][x:f:dif], color="tab:orange", label = "By")
#ax.plot(points_CADxyz[x:f:dif, 2], BFit[0][x:f:dif], color="tab:green", label = "Bx")

ax.set_xlabel('Z')
ax.set_ylabel('B field')
#ax.set_zlabel('Z (mm)')
#plt.title('HIDRA B-field points in Cartesian coordinates (+X: North, +Y: East)')
plt.tight_layout()
plt.legend()
#ax.view_init(elev=-157, azim=-44, roll=0)
plt.show()
#simIO.saveFig(OUTPUT_FILE_NAME+'.png', dpi=300)
"""