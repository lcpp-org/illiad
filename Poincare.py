import numpy as np
import scipy as sp
import os as os

from mesh import *
from coordtrans import *
from ode import blines, solvePoincare


##============================##
## DEFINE MESH AND LOAD FIELD ##
##============================##
b_hidra = CartesianField()
b_hidra.setToroidalGeometry(0.72, 0.19)

Bx, By, Bz = np.load('input_files/i3_hires_Bxyz.npy')
b_hidra.loadCartesianField(Bx, By, Bz)



##====================##
## SET UP FIELD LINES ##
##====================##
## NEED A BETTER WAY TO SET UP INITIAL POINTS!
Nlines = 14+13
spins = 500
length = (2*np.pi * b_hidra.R0) * spins

fl_R0 = np.array(np.linspace( -0.01, 0.120, Nlines))
fl_THETA0 = 0.0 * np.ones(Nlines)
fl_PHI0 = (np.pi/5.) * np.ones(Nlines)
#fl_THETA0 = np.array(np.linspace( 0.00, 0.00, Nlines))
#fl_PHI0 = np.array(np.linspace( np.pi/5., np.pi/5., Nlines))


ICs_RTP = np.transpose(np.vstack([fl_R0, fl_THETA0, fl_PHI0]))
print('Initial Conditions (RTP): ', ICs_RTP)

ICs_XYZ = np.zeros(shape=(Nlines, 3))
for i in range(Nlines):
    ICs_XYZ[i] = RTP_to_XYZ(ICs_RTP[i], b_hidra.R0)
#print('Initial Conditions (XYZ): ', ICs_XYZ)



##============##
## SET EVENTS ##
##============##
def inVV(t, p_XYZ, mesh):
    x, y, z = p_XYZ[:3]

    r = np.sqrt( x**2 + y**2 + z**2 + mesh.R0**2 - 2*mesh.R0*np.sqrt(x**2 + y**2) )
    return r - mesh.a
inVV.terminal = True

import phi_events
poincare_events = [inVV, 
                    phi_events.isphi9, 
                    phi_events.isphi18, 
                    phi_events.isphi27, 
                    phi_events.isphi36, 
                    phi_events.isphi45, 
                    phi_events.isphi54, 
                    phi_events.isphi63, 
                    phi_events.isphi72]


##===============================================##
## SOLVE FOR EACH INITIAL CONDITION CONCURRENTLY ##
##===============================================##
Poincare_output = [None]*Nlines

from functools import partial
import concurrent.futures
from time import perf_counter

t_start = perf_counter()
with concurrent.futures.ProcessPoolExecutor() as executor:
    solvePoincare_x = partial(solvePoincare, lineLength=length, field=b_hidra, solver_events=poincare_events)
    Poincare_output = executor.map(solvePoincare_x, ICs_XYZ)
t_stop = perf_counter()
elapsed_time = t_stop - t_start
print(f'## All Solvers Finished in {elapsed_time} seconds\n###############')

# convert 'generator' to 'list'
Poincare_output = list(Poincare_output)


## ============== ##
## POINCARE PLOTS ##
## ============== ##
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

UIUCcol = ('#13294B', '#FF5F0F', '#4D69A0', '#C84113')

phi_range = np.linspace( np.pi/20., (2/5)*np.pi, 8)
for n, phi_plot in enumerate(phi_range):
    #print('###########\n## PHI: ', phi_plot*(180/np.pi))
    print('## PHI: ', phi_plot*(180/np.pi))
    #print('###########')
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    #print('len of output: ', len(Poincare_output))
    for i in range(len(Poincare_output)):
        t_pts = Poincare_output[i][n+1] #skip wall event
        print(f'{len(t_pts)} points in Suface {i}')
        #print('t_pts:', t_pts)
        r_f = np.zeros(len(t_pts))
        th_f = np.zeros(len(t_pts))
        ph_f = np.zeros(len(t_pts))

        for j in range(len(t_pts)):
            #print('t_point ', t_pts[j])
            r_f[j], th_f[j], ph_f[j] = XYZ_to_RTP(t_pts[j][:3], b_hidra.R0)
        #print('phi at tpts: ', ph_f*(180./np.pi))

        f_output = np.array([th_f, r_f])
        np.save('Poincare_output_'+str(n)+'_'+str(i), f_output)

        #plt.scatter(th_f, r_f, s=0.1, c=UIUCcol[int(np.fmod(i,len(UIUCcol))
        plt.scatter(th_f, r_f, s=0.09)

    ax.set_rmax(b_hidra.a)
    plt.title(r'Poincare Plot, $\phi$={:02.0f}$\degree$'.format(phi_plot*180/np.pi))
    plt.savefig('Poincare_phi={:03.0f}.png'.format(phi_plot*180/np.pi),dpi=900)
plt.close('all')
