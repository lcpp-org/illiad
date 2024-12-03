## IMPORT
#import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import class_outputHandler as out
from mesh import *

## SET UP RUN DIRECTORY
simIO = out.IOHandler("+B_1q4_contours") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simIO.startLog()

## DEFINE MESH AND LOAD FIELD
Bx, By, Bz = np.load('input_files/i1q4_hires.npy')
mesh_prd = np.array([0, 1, 5], dtype=np.int32)
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(Bx, By, Bz, mesh_prd, errField=True)


mesh_ntheta = int(b_hidra.ntheta/2)
mesh_dtheta = b_hidra.dtheta*2
R     = np.linspace( b_hidra.r_min*2,       b_hidra.r_max,     int((b_hidra.nr//2)+1))
THETA = np.linspace( b_hidra.theta_min, b_hidra.theta_max, mesh_ntheta)
#PHI   = np.linspace( b_hidra.phi_min*2,     b_hidra.phi_max,   int(b_hidra.nphi/2))
PHI   = np.array([54,126,198,270,342])*(np.pi/180)#np.linspace( 9*(np.pi/180),     2*np.pi,   40)


#mesh_size = (b_hidra.nr, b_hidra.ntheta, b_hidra.nphi)
mesh_size = (R.size, THETA.size, PHI.size)

rr,tt = np.meshgrid(R,THETA)
rb,tb,pb = np.meshgrid(R,THETA,PHI)

# CALCULATE B-COMPONENTS #
Br = np.zeros(mesh_size)
Bpol = np.zeros(mesh_size)
Btor = np.zeros(mesh_size)
Bnorm = np.zeros(mesh_size)


for j, theta in enumerate(THETA):
	
	ctheta = np.cos(theta)
	stheta = np.sin(theta)

	for k, phi in enumerate(PHI):
		#print('theta, phi = {}, {}'.format(theta, phi))
		cphi = np.cos(phi)
		sphi = np.sin(phi)

		Xform = np.array([[ctheta*cphi, -ctheta*sphi, stheta],
						[ -stheta*cphi,  stheta*sphi, ctheta],
						[ -sphi, -cphi, 0]])

		for i, r in enumerate(R):
			bxyz, dum = b_hidra.interpField(np.asarray([r, theta, phi]))#, Cart=False)

			br, bpol, btor = np.dot(Xform, bxyz)
			#if r == 0.:
			if i == 0:
				bpol = 0

			Bnorm[i][j][k] = np.sqrt(bxyz[0]**2 + bxyz[1]**2 + bxyz[2]**2)
			Br[i][j][k] = br
			Bpol[i][j][k] = bpol
			Btor[i][j][k] = btor
print('Fields Calculated.')


def plot_Xsection(title, data, filename, phi_toPlot):
	print('Plotting ' + title + '...')
	max_data = np.max(data)
	min_data = np.min(data)
	contours = np.linspace(min_data, max_data, 24)

	# Adding endpoint for continuous plot through origin
	wrped_tt = np.concatenate((tt, tt[-1:] + mesh_dtheta))#b_hidra.dtheta
	wrped_rr = np.concatenate((rr, rr[-1:]))

	for i, p in enumerate(phi_toPlot):
		plot_data = np.transpose(data, [2,1,0])[i]
		loc_max = np.max(plot_data)
		#loc_min = np.min(plot_data)
	
		wrp_data = np.concatenate((plot_data, plot_data[0:1, :]), axis=0)

		fig = plt.figure()
		ax = fig.add_subplot(111, polar=True)
		plt.contourf(wrped_tt.T, wrped_rr.T, wrp_data.T, contours, cmap='viridis')

		ax.set_rmax(b_hidra.r_max)
		ax.set_rticks(np.arange(0.0, 0.19, 0.02))
		ax.yaxis.set_tick_params(labelsize=5)
		ax.grid(linewidth = 0.25, linestyle=':', c='k')

		plt.colorbar()
		plt.title(title + r', $\phi$={:3.0f}$\degree$ Max.={:.4f}'.format(p*180/np.pi, loc_max))

		#plt.savefig(filename + '_phi={:02.0f}.png'.format(p*180/np.pi),dpi=300)
		plot_name = filename + '_phi={:02.0f}.png'.format(p*180/np.pi)
		simIO.saveFig(plot_name)
	plt.close()

def getValuesAlong0(title, data,phi_toPlot):
	simIO.log.info("The values for {} are given for these radii \n {} \n ".format(title, R))
	theta0s = []
	for i, p in enumerate(phi_toPlot):
		plot_data = np.transpose(data, [2,1,0])[i]
		theta0 = plot_data[0]
		theta0s.append(theta0)
		simIO.log.info('{}\n at {}'.format(theta0, p*180/np.pi))
	
	fig = plt.figure()
	ax = fig.add_subplot()
	plt.title("{} diffferences".format(title))
	for i, line in enumerate(theta0s):
		ax.plot(R, line, label = "{}".format(PHI[i]*180/np.pi))
	plt.legend()
	plt.xticks(np.linspace(0, 0.19, 11))
	#plt.show()
	simIO.saveFig(title)
	plt.close()

getValuesAlong0("norm", Bnorm, PHI)
getValuesAlong0("radial", Br, PHI)
getValuesAlong0("poloidal", Bpol, PHI)
getValuesAlong0("toroidal", Bnorm, PHI)

'''
## NORM ##
plot_Xsection('B-field magnitude of HIDRA', Bnorm, 'Bnorm_HIDRA_i3ERR_hires', PHI)
## RADIAL ##
plot_Xsection('RADIAL B-field magnitude of HIDRA', Br, 'Bradial_HIDRA_i3ERR_hires', PHI)
### POLOIDAL ##
plot_Xsection('POLOIDAL B-field magnitude of HIDRA', Bpol, 'Bpoloidal_HIDRA_i3ERR_hires', PHI)
### TOROIDAL ##
plot_Xsection('TOROIDAL B-field magnitude of HIDRA', Btor, 'Btoroidal_HIDRA_i3ERR_hires', PHI)
'''

"""
## WALL PLOTS
## Re-Do with [r][theta][phi] !
Bnormplot = np.transpose(Bnorm, [2,0,1])[-1]
print(Bnormplot.shape)

THETA = np.linspace( -np.pi, np.pi, ntheta)
PHI = np.linspace( 0, 2*np.pi, (nphi-1)*5)

Bnormplot= np.roll(Bnormplot,int(len(THETA)/2),axis=1)
Bnormplot2 = Bnormplot[:-1,:]
print(Bnormplot2.shape)

Bnormplot3 = np.tile(Bnormplot2.T, 5)
print(Bnormplot3.shape)

phi, theta = np.meshgrid(np.degrees(PHI),np.degrees(THETA))
#phi, theta = np.meshgrid(np.degrees(PHI),np.degrees(THETA)[-15:4])

fig = plt.figure()
ax = fig.add_subplot(111, polar=False)
plt.contourf(np.transpose(phi),np.transpose(theta),Bnormplot3.T, contours, cmap='viridis')
plt.colorbar()
#plt.contour(np.transpose(phi),np.transpose(theta),Bnormplot2.T, contours, linewidths=0.2, colors='k' )
plt.xlabel('Toroidal Angle $\phi [\degree]$')
plt.xticks(np.arange(0,360, step=36))
plt.ylabel('Poloidal Angle $\theta [\degree]$')
plt.yticks(np.arange(-180,180, step=45))
plt.title(r'B-field magnitude of HIDRA (on wall)')
plt.savefig('HIDRA-i3_r=wall.png',dpi=300)
#plt.show()



Brplot = np.transpose(Br, [2,0,1])[-1]
print(Brplot.shape)

THETA = np.linspace( -np.pi, np.pi, ntheta)
PHI = np.linspace( 0, 2*np.pi, (nphi-1)*5)

Brplot= np.roll(Brplot,int(len(THETA)/2),axis=1)
Brplot2 = Brplot[:-1,:]
print(Brplot2.shape)

Brplot3 = np.tile(Brplot2.T, 5)
print(Brplot3.shape)

phi, theta = np.meshgrid(np.degrees(PHI), np.degrees(THETA))
#phi, theta = np.meshgrid(np.degrees(PHI),np.degrees(THETA)[-15:4])

fig = plt.figure()
ax = fig.add_subplot(111, polar=False)
plt.contourf(np.transpose(phi),np.transpose(theta),Brplot3.T, contoursr, cmap='viridis')
plt.colorbar()
#plt.contour(np.transpose(phi),np.transpose(theta),Brplot2.T, contours, linewidths=0.2, colors='k' )
plt.xlabel('Toroidal Angle $\phi [\degree]$')
plt.xticks(np.arange(0,360, step=36))
plt.ylabel('Poloidal Angle $\theta [\degree]$')
plt.yticks(np.arange(-180,180, step=45))
plt.title(r'Radial B-field magnitude of HIDRA (on wall)')
plt.savefig('Bradial-i3_r=wall.png',dpi=300)
"""