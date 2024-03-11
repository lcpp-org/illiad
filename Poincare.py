import numpy as np
import scipy as sp
import os as os

from mesh import *
from coordtrans import *
from interpolator import *
from ode import *



#Bx, By, Bz = np.load('~HIDRA\input_files\Bxyz_out.npy')
#path = 'input_files/'


Bx, By, Bz = np.load('input_files/i3_hires_Bxyz.npy')
b_hidra = Field()
b_hidra.setMeshValues(0.72, 0.19)
b_hidra.populateField(Bx, By, Bz)

## EVENTS
from phi_events import *

def inVV(t, p_XYZ, mesh):
    inVV.terminal = True
    x, y, z = p_XYZ[:3]

    r = np.sqrt( x**2 + y**2 + z**2 + mesh.R0**2 - 2*mesh.R0*np.sqrt(x**2 + y**2) )
    return r - mesh.a


def solvePoincare(init_conds, Nlines, lineLength, field):
    data = [None]*Nlines
        
    #mesh.XYZ_to_deltaWall.terminal = True
    #setattr(mesh.XYZ_to_deltaWall, terminal, True)
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
    
    poincare_events = [inVV, isphi0, isphi9, isphi18, isphi27, isphi36, isphi45, isphi54, isphi63, isphi72]
    #poincare_events = [isphi0, isphi9, isphi18, isphi27, isphi36, isphi45, isphi54, isphi63, isphi72]
    t_min = lineLength[0]
    t_max = 0.
    tloc = np.zeros(1)
    temp_size = int(init_conds[0].size)
        
    # Loop through number of fieldlines	
    for i in range(temp_size):
        print('Line #', str(i+1))
        Y0 = np.array([ fieldlines_X0[i], fieldlines_Y0[i], fieldlines_Z0[i], -fieldlines_direction[i]]) # ToDo: Use init_conds!!
        span = (0.0, lineLength[i])
        fieldlines = sp.integrate.solve_ivp(blines, span, Y0, args = ([field]),
                dense_output=False,
                events = poincare_events, 
                method='RK45', max_step=5e-4, rtol=1e-9, atol=1e-9) #3e-4
        print('List of Event 0: ', fieldlines.y_events[0])
        print('List of Event 2: ', fieldlines.y_events[2])
        data[i] = fieldlines.y_events
    return data, t_min, t_max



##=================================== ##
## SET UP FIELD LINES AND CALL SOLVER ##
##=================================== ##
## NEED A BETTER WAY TO SET UP INITIAL POINTS!
Nx = 3
Ny = 1
Nz = 1
spins = 5
Nlines = Nx*Ny*Nz

#fl_R0 = np.array(np.linspace( 0.120, 0.010, Nlines))
fl_R0 = np.array(np.linspace( 0.000, 0.090, Nlines))
fl_THETA0 = np.array(np.linspace( 0.00, 0.00, Nlines))
fl_PHI0 = np.array(np.linspace( np.pi/5., np.pi/5., Nlines))

fieldlines_X0 = toX(fl_R0, fl_THETA0, fl_PHI0, b_hidra.R0)
fieldlines_Y0 = toY(fl_R0, fl_THETA0, fl_PHI0, b_hidra.R0)
fieldlines_Z0 = toZ(fl_R0, fl_THETA0, fl_PHI0, b_hidra.R0)
fieldlines_direction = np.ones( Nlines )
fieldlines_length = np.ones( Nlines ) * 2*np.pi * b_hidra.R0 * spins

init_conds = np.array([ fieldlines_X0, fieldlines_Y0, fieldlines_Z0, -fieldlines_direction])



Poincare_output, tmin, tmax = solvePoincare(init_conds, Nlines, fieldlines_length, b_hidra)


## ============== ##
## POINCARE PLOTS ##
## ============== ##

## PLOTTING SETUP
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

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
    fl_phi = toPHI_nbless(jimx, jimy, jimz, b_hidra.R0)
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
            r_f[j] = toR_nbless(x_f, y_f, z_f, b_hidra.R0)
            th_f[j] = toTHETA_nbless(x_f, y_f, z_f, b_hidra.R0)
            ph_f[j] = toPHI_nbless(x_f, y_f, z_f, b_hidra.R0)
        print('phi at tpts: ', ph_f*(180./np.pi))
        
        f_output = np.array([th_f, r_f])
        np.save('Poincare_output_'+str(n)+'_'+str(i), f_output)
        
        #plt.scatter(th_f, r_f, s=0.1, c=UIUCcol[int(np.fmod(i,len(UIUCcol))
        plt.scatter(th_f, r_f, s=0.1)
        #endif
    
    ax.set_rmax(b_hidra.a)
    plt.title(r'Poincare Plot, $\phi$={:02.0f}$\degree$'.format(phi_plot*180/np.pi))
    plt.savefig('Poincare_phi={:03.0f}.png'.format(phi_plot*180/np.pi),dpi=900)
plt.close('all')
