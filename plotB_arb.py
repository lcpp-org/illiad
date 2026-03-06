## IMPORT
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# import classes.class_outputHandler as out
from classes.iohandler import IOHandler
from classes.mesh import Mesh
from utility.coordtrans import RTP_to_XYZ, XYZ_to_RTP
from magnetic_fiel_function_fitter import Magnetic_function_fitter
from sklearn.metrics import r2_score

# TOROIDAL AND HELICAL MAGNETIC FIELDS
TOROIDAL_CURRENT = 3.50 #[kA]
HELICAL_CURRENT = 6.30 #[kA]
CONFIG_TOR = 'default_toroidal'
CONFIG_HEL = 'default_helical'

DATA_FILE = 'input_files/Full_Magnetic_Field_Bounding_Box.csv'
#DATA_FILE = 'input_files/new_pfc_loc_coords.csv'
#DATA_FILE = 'input_files/large_box.csv'
#DATA_FILE = 'input_files/small_box.csv'

OUTPUT_DIRECTORY_NAME = "BFIELDS_093025"
OUTPUT_FILE_NAME = '35000-IT_6300-IH_full_box'

## SET UP RUN DIRECTORY
simIO = IOHandler(OUTPUT_DIRECTORY_NAME) 
simIO.startLog()
## DEFINE MESH AND LOAD FIELD
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
# Rotates the basis 180 CCW about Y-, then -18deg about Z-axis
phi_xform = np.radians(-18)
cosphi_xform = np.cos(phi_xform)
sinphi_xform = np.sin(phi_xform)
xFormMatrix = np.array([[-cosphi_xform, -sinphi_xform, 0.0],
                       [-sinphi_xform,  cosphi_xform, 0.0],
                       [0.0,           0.0,          -1.0]])

# Rotates the basis 27 degrees CW around the Z axis
phi_plate = np.radians(27)
cosphi_plate = np.cos(phi_plate)
sinphi_plate = np.sin(phi_plate)
plateMatrix = np.array([[cosphi_plate,  sinphi_plate,  0.0],
                        [-sinphi_plate,  cosphi_plate,   0.0],
                        [0.0,                    0.0,   1.0]])

#inclination of the plate, high at the high field side and low at the low field side
theta_plate = np.radians(5)
costheta_plate = np.cos(theta_plate)
sintheta_plate = np.sin(theta_plate)
thetaMatrix = np.array([[costheta_plate, 0, sintheta_plate],
                        [0, 1, 0],
                        [-sintheta_plate, 0, costheta_plate]])

#convert to RTP coordinates using coordtrans.
points_SIMxyz = np.zeros_like(points_CADxyz)
fields_SIMxyz = np.zeros_like(points_CADxyz)
fields_CADxyz = np.zeros_like(points_CADxyz)

fields_PLATExyz = np.zeros_like(points_CADxyz)
points_PLATExyz = np.zeros_like(points_CADxyz)
fields_FLUIDxyz = np.zeros_like(points_CADxyz)
points_FLUIDxyz = np.zeros_like(points_CADxyz)

for i, point_cad in enumerate(points_CADxyz):
    points_SIMxyz[i] = np.dot(xFormMatrix, point_cad/1000)  #convert from mm to m
    
    points_PLATExyz[i] = np.dot(plateMatrix, point_cad)
    points_FLUIDxyz[i] = np.dot(thetaMatrix, points_PLATExyz[i])
    
    fields_SIMxyz[i] = b_hidra.interpField(points_SIMxyz[i], Cart=True)[0]
    fields_CADxyz[i] = np.dot(xFormMatrix, fields_SIMxyz[i])  # apply inverse rotation
    
    fields_PLATExyz[i] = np.dot(plateMatrix, fields_CADxyz[i]) 
    fields_FLUIDxyz[i] = np.dot(thetaMatrix, fields_PLATExyz[i])
#print(f'{points_SIMxyz[:, 0]=}')

#save output as a csv with header x,y,z,bx,by,bz and data from points_CADxyz and fields_PLATExyz
output_data = np.hstack((points_CADxyz, fields_PLATExyz))
simIO.saveCSV(output_data, OUTPUT_FILE_NAME+'.csv', header='x,y,z,bx,by,bz')

def fit():
    test = Magnetic_function_fitter(f"output/{OUTPUT_DIRECTORY_NAME}/data/{OUTPUT_FILE_NAME}.csv")
    fields = ['bx', 'by', 'bz']
    BFit = []
    for i in fields:
        #print(i)
        deg = 2
        cart_cord, X_poly, fit_coeffs = test.fitter(deg, i)
        #test.fit_tester(fit_bx, i)
        print(f"The R2 score is {r2_score(np.dot(X_poly, fit_coeffs), test.magnetic_data[i]):0.5f}")
        #BFit.append(fit_b)
        BFit.append(np.dot(X_poly, fit_coeffs))
        print(f"Equation for {i}: {fit_coeffs[0]:0.2e} + {fit_coeffs[1]:0.2e}(x+x0) + {fit_coeffs[2]:0.2e}(y+y0) + {fit_coeffs[3]:0.2e}(z+z0) + {fit_coeffs[4]:0.2e}(x+x0)^2 + {fit_coeffs[5]:0.2e}(x+x0)(y+y0) + {fit_coeffs[6]:0.2e}(x+x0)(z+z0) + {fit_coeffs[7]:0.2e}(y+y0)^2 + {fit_coeffs[8]:0.2e}(y+y0)(z+z0) + {fit_coeffs[9]:0.2e}(z+z0)^2")
    BFit = np.array(BFit)
fit()

def points(): 
    # plot the points in 3D, with the markers colored by the B-field magnitude
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #sc = ax.scatter(*points_SIMxyz.T*1000, c=np.linalg.norm(fields_SIMxyz, axis=1), cmap='viridis', marker='o')
    sc = ax.scatter(*points_CADxyz.T, c=np.linalg.norm(fields_CADxyz, axis=1), cmap='viridis', marker='o')
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
        magnitudes.append(np.sqrt(Bx**2 + By**2 + Bz**2))
    #print(magnitudes)
    colormap = cm.viridis
    norm = Normalize()
    norm.autoscale(magnitudes)
    colors = colormap(norm(magnitudes))

    # plot the points in 3D, with the vectors pointing in the direction of the B-field
    zpoints = 1
    for j in range(zpoints):
        slice_FLUIDxyz = points_FLUIDxyz[j::zpoints]
        colors_FLUID = colors[j::zpoints]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, point in enumerate(slice_FLUIDxyz):
            x_0,y_0,z_0 = point
            Bx, By, Bz = fields_FLUIDxyz[i]
            Bstrength = 0.02*np.sqrt(Bx**2 + By**2 + Bz**2)
            ax.quiver(x_0, y_0, (Bx/Bstrength), (By/Bstrength), color=colors_FLUID[i])
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        #ax.set_zlabel('Z (mm)')
        plt.title('HIDRA B-field points in FLUID coordinates (+X: width of plate, +Y: lenght of plate)')
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