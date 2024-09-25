import numpy as np
import numba as nb

import matplotlib.pyplot as plt
from matplotlib import patches

import class_outputHandler as out
from mesh import *
from coordtrans import *
from anlys_funcs import *
from poincare_gen import Gen_Poincare
from point_generators import generateSeedShells
from particle import *


def plot_Xsection(mesh, title, data, filename, phi_toPlot):
	print('Plotting ' + title + '...')
	max_data = np.max(data)
	min_data = np.min(data)
	contours = np.linspace(min_data, max_data, 9)

	for i, p in enumerate(phi_toPlot):
		plot_data = np.transpose(data, [2,1,0])[i]
		loc_max = np.max(plot_data)
		loc_min = np.min(plot_data)
	
		wrped_tt = np.concatenate((tt, tt[-1:] + b_hidra.dtheta))
		wrped_rr = np.concatenate((rr, rr[-1:]))
		wrp_data = np.concatenate((plot_data, plot_data[0:1, :]), axis=0)

		fig = plt.figure()
		ax = fig.add_subplot(111, polar=True)
		plt.contourf(np.transpose(wrped_tt), np.transpose(wrped_rr), np.transpose(wrp_data), contours, cmap='viridis')

		ax.set_rmax(b_hidra.a)
		ax.set_rticks(np.arange(0.0, 0.19, 0.02))
		ax.yaxis.set_tick_params(labelsize=5)
		ax.grid(linewidth = 0.25, linestyle=':', c='k')

		plt.colorbar()
		plt.title(title + r', $\phi$={:3.0f}$\degree$ Max.={:.4f}'.format(p*180/np.pi, loc_max))

		#plt.savefig(filename + '_phi={:02.0f}.png'.format(p*180/np.pi),dpi=300)
		plot_name = filename + '_phi={:02.0f}.png'.format(p*180/np.pi)
		simIO.saveFig(plot_name)
	plt.close()


## SET UP RUN DIRECTORY
simIO = out.IOHandler("HIDRA_1q4ERR_1500s") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simIO.startLog()

## DEFINE MESH AND LOAD FIELD
BX, BY, BZ = np.load('input_files/HIDRA_i4ERR_hires.npy')
mesh_prd = np.array([0, 1, 5], dtype=np.int32)
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(BX, BY, BZ, mesh_prd, errField=True)

simIO.log.info('Start PSI CALCULATION\n')
b_hidra.calculate_psi()


simIO.log.info('BEGIN PSI PLOTTING\n')
R     = np.linspace( 0.0,           b_hidra.a, b_hidra.nr)
THETA = np.linspace( 0.0,           2*np.pi/1, b_hidra.ntheta)
PHI   = np.linspace( b_hidra.dphi,  2*np.pi/5, b_hidra.nphi)

rr,tt = np.meshgrid(R,THETA)
rb,tb,pb = np.meshgrid(R,THETA,PHI)


## PLOT PSI
plot_Xsection(b_hidra, 'B-field magnitude of HIDRA', b_hidra.PSI_ideal, 'idealPSI_iq4', PHI)