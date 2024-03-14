import numpy as np
import scipy as sp

from mesh import *
from coordtrans import *
from ode import blines, solvePoincare
import class_outputHandler as out


##============================##
##   SET UP OUTPUT FOLDERS    ##
## (GIVER YOUR OUTPUT A NAME) ##
##============================##
# right now, data and plots WILL be overwritten if the directory already exists!
simOut = out.outputHandler("24lines_500spins")
simOut.startLog()


##============================##
## DEFINE MESH AND LOAD FIELD ##
##============================##
b_hidra = CartesianField()
b_hidra.setToroidalGeometry(0.72, 0.19)
b_hidra.loadCartesianField_fromFile('i3_hires_Bxyz.npy')


##====================##
## SET UP FIELD LINES ##
##====================##
## NEED A BETTER WAY TO SET UP INITIAL POINTS!
Nlines = 24#41
spins = 500
length = (2*np.pi * b_hidra.R0) * spins

fl_R0 = np.array(np.linspace( 0.120, 0.028, Nlines))
fl_THETA0 = 0.0 * np.ones(Nlines)
fl_PHI0 = (np.pi/5.) * np.ones(Nlines)

ICs_RTP = np.transpose(np.vstack([fl_R0, fl_THETA0, fl_PHI0]))

ICs_XYZ = np.zeros(shape=(Nlines, 3))
for i in range(Nlines):
    ICs_XYZ[i] = RTP_to_XYZ(ICs_RTP[i], b_hidra.R0)

simOut.log.info(f'Initial Conditions (RTP): {ICs_RTP}')    
simOut.log.debug('Initial Conditions (XYZ): ', ICs_XYZ)


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
    solvePoincare_x = partial(solvePoincare, maxLength=length, field=b_hidra, solver_events=poincare_events)
    Poincare_output = executor.map(solvePoincare_x, ICs_XYZ)
t_stop = perf_counter()
elapsed_time = t_stop - t_start
simOut.log.info(f'ALL SOLVERS FINISHED IN {elapsed_time} seconds\n###############\n\n')

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
#from cycler import cycler
#UIUCcol = ['#13294B', '#FF5F0F', '#4D69A0', '#C84113']

phi_range = np.linspace( np.pi/20., (2/5)*np.pi, 8)
for n, phi_plot in enumerate(phi_range):
    simOut.log.info(f'## PHI: {phi_plot*(180/np.pi)}')

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    for i in range(len(Poincare_output)):
        t_pts = Poincare_output[i][n+1] #skip wall event

        r_f = np.zeros(len(t_pts))
        th_f = np.zeros(len(t_pts))
        ph_f = np.zeros(len(t_pts))

        for j in range(len(t_pts)):
            r_f[j], th_f[j], ph_f[j] = XYZ_to_RTP(t_pts[j][:3], b_hidra.R0)

        f_output = np.array([th_f, r_f])
        fname = 'Poincare_output_'+str(n)+'_'+str(i)
        simOut.saveNumpyData(f_output, fname)

        plt.scatter(th_f, r_f, marker='.', s=1.5, linewidths=0.0)

        simOut.log.debug(f'\t{len(t_pts)} points in Suface {i}')
        simOut.log.debug(f'phi (deg.) at tpts: {ph_f*(180./np.pi)}')        

    ax.set_rmax(b_hidra.a)
    plt.title(r'Poincare Plot, $\phi$={:02.0f}$\degree$'.format(phi_plot*180/np.pi))

    plot_name = 'Poincare_phi={:03.0f}.png'.format(phi_plot*180/np.pi)
    simOut.saveFig(plot_name)

plt.close('all')

simOut.log.info('## SIM FINISHED ##\n\n\n\n')