##########
## IMPORTS
##########
from math import copysign
from math import fmod
import scipy as sp
import numpy as np
from mesh import *
from numpy import ndarray as a

from numba import jit, prange
import numba as nb

from coordtrans import *

Rmaj = 0.72 #[m]
Rmin = 0.19 #[m]

## LOAD DATA, CREATE GRID
Bx,By,Bz = np.load('Bxyz_out.npy')
Bnorm = np.load('Bnorm_out.npy')

# Generate uniform grid of points along (r,th,phi) coordinates
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
