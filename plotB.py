## IMPORT
#import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

Rmaj = 0.72 #[m]
Rmin = 0.19 #[m]

Bx,By,Bz = np.load('input_files/Bxyz_iota-1q3_MAXPOWER_hires.npy')
Bnorm = np.load('input_files/Bnorm_iota-1q3_MAXPOWER_hires.npy')
print('Bnorm.shape={}'.format(Bnorm.shape))
nr, ntheta, nphi = Bnorm.shape

theta_periods = 1
phi_periods = 5

theta_maximum = 2*np.pi / theta_periods
phi_maximum = 2*np.pi / phi_periods

dtheta = theta_maximum/ntheta
dphi = phi_maximum/nphi

R     = np.linspace( 0.0,                     Rmin,     nr)
THETA = np.linspace( dtheta, 2*np.pi/theta_periods, ntheta)
PHI   = np.linspace( dphi,     2*np.pi/phi_periods,   nphi)

rr,tt = np.meshgrid(R,THETA)
rb,tb,pb = np.meshgrid(R,THETA,PHI)

### assuming s_phi = s_theta = 1:
### (not what fieldlines uses! (uses s_phi = -1))
###e_rad = [cos(THETA)*cos(PHI),
###		 cos(THETA)*sin(PHI),
###		 sin(THETA)]
###
###e_thet = [-sin(THETA)*cos(PHI),
###		  -sin(THETA)*sin(PHI),
###		  cos(THETA)]
###
###e_phi = [-sin(PHI),
###		 cos(PHI,
###		 0)]

# CALCULATE B-COMPONENTS #
Br = np.zeros(Bnorm.shape)
Bpol = np.zeros(Bnorm.shape)
Btor = np.zeros(Bnorm.shape)
for i in range(0, R.size):
	for j in range(0, THETA.size):
		for k in range(0, PHI.size):

			Br[i][j][k]   = Bx[i][j][k]*np.cos(THETA[j])*np.cos(PHI[k]) - By[i][j][k]*np.cos(THETA[j])*np.sin(PHI[k]) + Bz[i][j][k]*np.sin(THETA[j])

			if R[i] == 0.:
				Bpol[i][j][k] = 0
			else:
				Bpol[i][j][k] = (-1)*Bx[i][j][k]*np.sin(THETA[j])*np.cos(PHI[k]) + By[i][j][k]*np.sin(THETA[j])*np.sin(PHI[k]) + Bz[i][j][k]*np.cos(THETA[j])

			Btor[i][j][k] = (-1)*Bx[i][j][k]*np.sin(PHI[k]) - By[i][j][k]*np.cos(PHI[k])
print(Bnorm.shape)


def plot_Xsection(title, data, filename, phi_toPlot):
	print('Plotting ' + title + '...')
	max_data = np.max(data)
	min_data = np.min(data)
	contours = np.linspace(min_data, max_data, 9)

	for i, p in enumerate(phi_toPlot):
		plot_data = np.transpose(data, [2,1,0])[i]
		loc_max = np.max(plot_data)
		loc_min = np.min(plot_data)
	
		wrped_tt = np.concatenate((tt, tt[-1:] + dtheta))
		wrped_rr = np.concatenate((rr, rr[-1:]))
		wrp_data = np.concatenate((plot_data, plot_data[0:1, :]), axis=0)

		fig = plt.figure()
		ax = fig.add_subplot(111, polar=True)
		plt.contourf(np.transpose(wrped_tt), np.transpose(wrped_rr), np.transpose(wrp_data), contours, cmap='viridis')

		ax.set_rmax(Rmin)
		ax.set_rticks(np.arange(0.0, 0.19, 0.02))
		ax.yaxis.set_tick_params(labelsize=5)
		ax.grid(linewidth = 0.25, linestyle=':', c='k')

		plt.colorbar()
		plt.title(title + r', $\phi$={:3.0f}$\degree$ Max.={:.4f}'.format(p*180/np.pi, loc_max))

		plt.savefig(filename + '_phi={:02.0f}.png'.format(p*180/np.pi),dpi=300)
	plt.close()


## NORM ##
plot_Xsection('B-field magnitude of HIDRA', Bnorm, 'Bnorm_i3-MaxPower', PHI)
## RADIAL ##
plot_Xsection('RADIAL B-field magnitude of HIDRA', Br, 'Bradial_i3-MaxPower', PHI)
### POLOIDAL ##
plot_Xsection('POLOIDAL B-field magnitude of HIDRA', Bpol, 'Bpoloidal_i3-MaxPower', PHI)
### TOROIDAL ##
plot_Xsection('TOROIDAL B-field magnitude of HIDRA', Btor, 'Btoroidal_i3-MaxPower', PHI)


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