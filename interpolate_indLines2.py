##########
## IMPORTS
##########
from math import copysign
from math import fmod
import scipy as sp
import numpy as np
from numpy import ndarray as a

from numba import jit, prange
import numba as nb

Rmaj = 0.72 #[m]
Rmin = 0.19 #[m]

## LOAD DATA, CREATE GRID
Bx,By,Bz = np.load('Bxyz_out.npy')
Bnorm = np.load('Bnorm_out.npy')

# Generate uniform grid of points along (r,phi,th) coordinates
#
# Coordinates
#
#       r : [m] radial coordinate
#       theta : [rad] poloidal angle
#       phi : [rad] toroidal angle
#
# Ranges 
#
#       0.0 < r < Rmin
#       0.0 < theta < 2*pi (full poloidal extension)
#       0.0 < phi < 2*pi (full toroidal extension)
nphi, ntheta, nr = Bx.shape
dr = Rmin/(nr-1)
dtheta = 2*np.pi/(ntheta-1)
dphi = 2*np.pi/(nphi-1)

R = np.linspace( 0.0, Rmin, nr)
THETA = np.linspace( 0, 2*np.pi, ntheta)
PHI = np.linspace( 0, 2*np.pi, nphi)


####################
## DEFINE FUNCTIONS:
####################

## TRANSFORM TO CARTESIAN COORDINATES
#############
#@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toX(r, theta, phi, Rmajor):
	return (Rmajor + r*np.cos(theta))*np.cos(phi)
#@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toY(r, theta, phi, Rmajor):
	return (Rmajor + r*np.cos(theta))*np.sin(phi)
#@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toZ(r, theta, phi, Rmajor):
	return r*np.sin(theta)

## TRANSFORM TO TOROIDAL COORDINATES
#############
@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toR(x, y, z, Rmajor):
	return np.sqrt( x**2 + y**2 + z**2 + Rmajor**2 - 2*Rmajor*np.sqrt(x**2 + y**2) )
@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toTHETA(x, y, z, Rmajor):
	den = np.sqrt(x**2 + y**2) - Rmajor
	temp = np.arctan2(z,den)
	temp2 = np.where(temp<0, 2*np.pi+temp, temp)
	return temp2.item()
@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toPHI(x, y, z, Rmajor):
	temp = np.arctan2(y,x)
	temp2 = np.where(temp<0, 2*np.pi+temp, temp)
	return temp2.item()


## FIND THE 8 CORNER POINTS OF THE CELL CONTAINING THE PARTICLE
## RETURN AS A LIST OF TUPLES
#############
@jit(nb.types.Array(nb.int32, 2, "C")
(nb.types.Array(nb.float64, 1, "C"), nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True)), nopython=True)
def findNodes(point, r, theta, phi):
	th_loc = np.float64
	ph_loc = np.float64
	
	r_loc = float(point[0])
	th_loc = np.fmod(point[1], (2*np.pi)) # periodic boundary
	ph_loc = np.fmod(point[2], (2*np.pi)) # periodic boundary
	if point[0] > Rmin:
		print('POINT OUTSIDE OF MESH!')
		rlb = nr-1
		thlb = ntheta-1 
		phlb = nphi-1
	else:
		#temp1 = np.where(r_loc < (r + dr)) 
		#rlb = temp1[0][0]
		
		#rlb = r_loc // dr
		rlb = np.floor(r_loc/dr)

		#temp2 = np.where(th_loc < (theta + dtheta))
		#thlb = temp2[0][0]
		
		#thlb = th_loc // dtheta
		thlb = np.floor(th_loc/dtheta)
		#thlb = np.fmod(thlb, ntheta)
		
		#temp3 = np.where(ph_loc < (phi + dphi))
		#phlb = temp3[0][0]
		
		#phlb = ph_loc // dphi
		phlb = np.floor(ph_loc/dphi)
		#phlb = np.fmod(phlb, nphi)

	nodeOut = np.array(
			[(rlb, thlb, phlb),                                              (rlb+1, thlb, phlb),
			(rlb, np.fmod((thlb+1),len(theta)), phlb),                       (rlb+1, np.fmod((thlb+1),len(theta)), phlb),
			(rlb, thlb, np.fmod((phlb+1),len(phi))),                         (rlb+1, thlb, np.fmod((phlb+1),len(phi))),
			(rlb, np.fmod((thlb+1),len(theta)), np.fmod((phlb+1),len(phi))), (rlb+1, np.fmod((thlb+1),len(theta)), np.fmod((phlb+1),len(phi)))],
			dtype=np.int32)
			
	return nodeOut

## CALCULATE THE VOLUME OF THE CELL DEFINED BY TWO OPPOSITE CORNERS
#############
@jit(nb.float64(nb.types.Array(nb.float64, 1, "C"), nb.types.Array(nb.float64, 1, "C")), nopython=True)
def volume(point1, point2):
	#return abs(((Rmaj/2)*(point2[0]**2 - point1[0]**2) + (1/3)*(point2[0]**3 - point1[0]**3)) * (point2[1] - point1[1])*(point2[2] - point1[2])) # using small angle theorem: sin(x)~x
	return abs(  (Rmaj/2)*(point2[0]**2 - point1[0]**2)*(point2[1] - point1[1])*(point2[2] - point1[2]) + (1/3)*(point2[0]**3 - point1[0]**3)*np.sin(point2[1] - point1[1])*(point2[2] - point1[2]) )

## INTERPOLATE THE VALUES OF THE MAGNETIC FIELDS AT THE PARTICLE LOCATION
#############
@jit(nb.types.Array(nb.float64, 1, "C")
(nb.types.Array(nb.float64, 1, "C"),
nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True),
nb.types.Array(nb.float64, 3, "C", readonly=True),nb.types.Array(nb.float64, 3, "C", readonly=True),nb.types.Array(nb.float64, 3, "C", readonly=True)),
nopython=True)
def interpField(point, r, theta, phi, bx, by, bz):
	vols = np.zeros(8)
	t_bx = 0.
	t_by = 0.
	t_bz = 0.
	
	point_tor = np.array([toR(point[0], point[1], point[2], Rmaj), toTHETA(point[0], point[1], point[2], Rmaj), toPHI(point[0], point[1], point[2], Rmaj)])
	cellpts = findNodes(point_tor, r, theta, phi)
	for j in range(8):
		thetaj = np.fmod(cellpts[j][1],len(THETA))
		phij = np.fmod(cellpts[j][2],len(PHI))
		# fmod already done in 'findNodes'
		#cpoint = np.array([R[cellpts[j][0]], 
		#					THETA[cellpts[j][1]],
		#					PHI[cellpts[j][2]]])
		cpoint = np.array([R[cellpts[j][0]], 
							THETA[thetaj],
							PHI[phij]]) #modulus on periodic boundaries
		vols[j] = volume(point_tor, cpoint)
		t_bx += vols[j] * Bx[cellpts[j][2], cellpts[j][1], cellpts[j][0]]
		t_by += vols[j] * By[cellpts[j][2], cellpts[j][1], cellpts[j][0]]
		t_bz += vols[j] * Bz[cellpts[j][2], cellpts[j][1], cellpts[j][0]]
	t_bx = t_bx/sum(vols)
	t_by = t_by/sum(vols)
	t_bz = t_bz/sum(vols)
	t_b = np.array([t_bx,t_by,t_bz])

	return t_b


## ==================================== ##-
## FIELD LINE SOLVER (FROM "Bfield.py") ##
## ==================================== ##
@jit(nb.types.Array(nb.float64, 1, "C")(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def blines(t,y):
	B = np.zeros((3,1))
	X=y[0]
	Y=y[1]
	Z=y[2]
	direction=y[3]
	point = np.array([ X, Y, Z ])

	B = interpField(point, R, THETA, PHI, Bx, By, Bz)
	Bnorm = np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
	dY    = np.zeros(4)
	dY[0] = direction * B[0]/Bnorm
	dY[1] = direction * B[1]/Bnorm
	dY[2] = direction * B[2]/Bnorm
	dY[3] = 0.0
	return dY


## ====== ##
## EVENTS ##
## ====== ##
## DETERMINE WHETHER POINT IS WITHIN VACUUM VESSEL
## >0, WITHIN VESSEL, <=0, OUTSIDE VESSEL
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def inVV(t, point):
	return Rmin - toR(point[0],point[1], point[2],  Rmaj) - 0.0001
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi0(t, point):
	return 0.*(np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi9(t, point):
	return 1. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi18(t, point):
	return 2. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi27(t, point):
	return 3. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi36(t, point):
	return 4. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi45(t, point):
	return 5. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi54(t, point):
	return 6. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi63(t, point):
	return 7. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi72(t, point):
	return 8. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)


## (NOT?) PARALLELIZABLE FUNCTION FOR NUMBA (Lot's of Field Lines!)
#@jit(parallel=True)#, nogil=True)
def solvePoincare(init_conds, Nlines, lineLength):
	data = [None]*Nlines
	inVV.terminal = True
	isphi0.direction = -1.0
	isphi9.direction = -1.0
	isphi18.direction = -1.0
	isphi27.direction = -1.0
	isphi36.direction = -1.0
	isphi45.direction = -1.0
	isphi54.direction = -1.0
	isphi63.direction = -1.0
	isphi72.direction = -1.0
	t_min = lineLength[0]
	t_max = 0.
	tloc = np.zeros(1)
	temp_size = int(init_conds[0].size)
	# Loop through number of fieldlines	
	#for i in prange(temp_size):
	for i in range(temp_size):
		print('Line #', str(i+1))
		Y0 = np.array([ fieldlines_X0[i], fieldlines_Y0[i], fieldlines_Z0[i], -fieldlines_direction[i]]) # ToDo: Use init_conds!!
		span = (0.0, lineLength[i])
		fieldlines = sp.integrate.solve_ivp(blines, span, Y0,
						dense_output=False,
						events = [inVV, isphi0, isphi9, isphi18, isphi27, isphi36, isphi45, isphi54, isphi63, isphi72], 
						method='RK45', max_step=5e-4, rtol=1e-9, atol=1e-9) #3e-4
		print('Wall Events: ', fieldlines.y_events[0])
		data[i] = fieldlines.y_events
	return data, t_min, t_max


##=================================== ##
## SET UP FIELD LINES AND CALL SOLVER ##
##=================================== ##
## NEED A BETTER WAY TO SET UP INITIAL POINTS!
Nx = 31
Ny = 1
Nz = 1
spins = 1000
Nlines = Nx*Ny*Nz

#fl_R0 = np.array(np.linspace( 0.070, 0.070, Nlines))
#fl_THETA0 = np.array(np.linspace( 0.00, 0.00, Nlines))
#fl_PHI0 = np.array(np.linspace( np.pi/5., np.pi/5., Nlines))

fl_R0 = np.array(np.linspace( 0.120, 0.010, Nlines))
#fl_R0 = np.array(np.linspace( 0.060, 0.080, Nlines))
fl_THETA0 = np.array(np.linspace( 0.00, 0.00, Nlines))
fl_PHI0 = np.array(np.linspace( np.pi/5., np.pi/5., Nlines))


fieldlines_X0 = toX(fl_R0, fl_THETA0, fl_PHI0, Rmaj)
fieldlines_Y0 = toY(fl_R0, fl_THETA0, fl_PHI0, Rmaj)
fieldlines_Z0 = toZ(fl_R0, fl_THETA0, fl_PHI0, Rmaj)
fieldlines_direction = np.ones( Nlines )
fieldlines_length = np.ones( Nlines ) * (2*np.pi*Rmaj)*spins

init_conds = np.array([ fieldlines_X0, fieldlines_Y0, fieldlines_Z0, -fieldlines_direction])
#inVV.terminal = True
Poincare_output, tmin, tmax = solvePoincare(init_conds, Nlines, fieldlines_length)


##################
## PLOTTING SETUP
##################
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

#import matplotlib.style as mplstyle
#mplstyle.use(['dark_background', 'fast'])

plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

##############
"""## PLOTTING 3D
###############
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
## INITIALIZE FIGURE
fig = plt.figure(figsize=(12.8, 9.6)) #default: 6.4 x 4.8
ax = fig.add_subplot(projection='3d')

## PLOT VACUUM VESSEL TORUS
vvres = 100
# theta: poloidal angle; phi: toroidal angle
ptheta = np.linspace(np.pi, 2*np.pi, vvres)
pphi   = np.linspace(0, 2.*np.pi, vvres)
ptheta, pphi = np.meshgrid(ptheta, pphi)

px = (Rmaj + Rmin*np.cos(ptheta)) * np.cos(pphi)
py = (Rmaj + Rmin*np.cos(ptheta)) * np.sin(pphi)
pz = Rmin * np.sin(ptheta)
ax.set_zlim(-0.7,0.7)
#ax.plot_surface(px, py, pz, rstride=3, cstride=3, color='grey', edgecolor='k', linewidth=0.1, alpha=0.3, shade=True) #'dimgrey'
ax.plot_wireframe(px, py, pz, color='k', linewidth=0.5, rstride=20, cstride=20,)

## PLOT FIELDLINES
for i in range(np.size(output)):
	fieldlines1 = output[i]
	if fieldlines1.t_events[0].size == 0: #t_events is empty if the fieldLine remains confined to the Vacuum Vessel
		ax.plot( fieldlines1.y[0], fieldlines1.y[1], fieldlines1.y[2], linewidth=1.)
plt.title('Fieldlines of HIDRA')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.margins(0.05)

plt.savefig('FieldLine.png', bbox_inches='tight', dpi=600)
plt.show()
"""

## ============== ##
## POINCARE PLOTS ##
## ============== ##
## Define functions without numba @jit decorations
def toR_nbless(x, y, z, Rmajor):
	return np.sqrt( x**2 + y**2 + z**2 + Rmajor**2 - 2*Rmajor*np.sqrt(x**2 + y**2) )
def toTHETA_nbless(x, y, z, Rmajor):
	den = np.sqrt(x**2 + y**2) - Rmajor
	temp = np.arctan2(z,den)
	temp2 = np.where(temp<0, 2*np.pi+temp, temp)
	return temp2.item()
def toPHI_nbless(x, y, z, Rmajor):
	temp = np.arctan2(y,x)
	temp2 = np.where(temp<0, 2*np.pi+temp, temp)
	return temp2.item()
def fieldline_phi(t, target_phi, fl_xyz):
	jimx, jimy, jimz, temp = fl_xyz(t)
	fl_phi = toPHI_nbless(jimx, jimy, jimz, Rmaj)
	return fl_phi - target_phi

UIUCcol = ('#13294B', '#FF5F0F', '#4D69A0', '#C84113')

phi_range = np.linspace(0., (2/5)*np.pi, 9)
for n, phi_plot in enumerate(phi_range):
	print('###########\n## PHI: ', phi_plot*(180/np.pi))
	print('###########')
	fig = plt.figure()
	ax = fig.add_subplot(111, polar=True)
	
	#print('^%^%^: ', len(Poincare_output))
	for i in range(len(Poincare_output)):
		t_pts = Poincare_output[i][n+1] #skip wall event
		#print('t_pts:', t_pts)
		r_f = np.zeros(len(t_pts))
		th_f = np.zeros(len(t_pts))
		ph_f = np.zeros(len(t_pts))

		for j in range(len(t_pts)):
			#print('t_point ', t_pts[j])
			x_f = t_pts[j][0]
			y_f = t_pts[j][1]
			z_f = t_pts[j][2]
			r_f[j] = toR_nbless(x_f, y_f, z_f, Rmaj)
			th_f[j] = toTHETA_nbless(x_f, y_f, z_f, Rmaj)
			ph_f[j] = toPHI_nbless(x_f, y_f, z_f, Rmaj)
		print('phi at tpts: ', ph_f*(180./np.pi))
		
		f_output = np.array([th_f, r_f])
		np.save('Poincare_output_'+str(n)+'_'+str(i), f_output)
		
		#plt.scatter(th_f, r_f, s=0.1, c=UIUCcol[int(np.fmod(i,len(UIUCcol))
		plt.scatter(th_f, r_f, s=0.1)
		#endif
	
	ax.set_rmax(Rmin)
	plt.title(r'Poincare Plot, $\phi$={:02.0f}$\degree$'.format(phi_plot*180/np.pi))
	plt.savefig('Poincare_phi={:03.0f}.png'.format(phi_plot*180/np.pi),dpi=900)
plt.close('all')

###################
##  _____        ##
## |_   _|       ##
##   | |         ##
##  _| |_        ##
## |_____LLINOIS ##
###################
