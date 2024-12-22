## IMPORT
#import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import class_outputHandler as out
from mesh import *

'''
Things to change include
simIO out
input magnetic file to be loaded
angles for PHI
booleans for highToLow and deltas for getValuesAlong0
plot_XSection (comment out or not)

'''
## SET UP RUN DIRECTORY
simIO = out.IOHandler("Bfield_graphs/1q3_contours_6PFC_deltas_max") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simIO.startLog()

## DEFINE MESH AND LOAD FIELD
Bx, By, Bz = np.load('input_files/i1q3_hires_max.npy')
mesh_prd = np.array([0, 1, 5], dtype=np.int32)
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(Bx, By, Bz, mesh_prd, errField=True)


mesh_ntheta = int(b_hidra.ntheta/2)
mesh_dtheta = b_hidra.dtheta*2
R     = np.linspace( b_hidra.r_min,       b_hidra.r_max,    int((b_hidra.nr//2)+1))
THETA = np.linspace( b_hidra.theta_min, b_hidra.theta_max, mesh_ntheta)
#PHI   = np.linspace( b_hidra.phi_min,     b_hidra.phi_max,   int(b_hidra.nphi/2))
PHI   = np.array([18, 45, 54, 117, 189, 261, 333])*(np.pi/180)#np.linspace( 9*(np.pi/180),     2*np.pi,   40)


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
			bxyz, dum = b_hidra.interpField(np.asarray([r, theta, phi]), Cart=False)

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

def calcMaxDifference(theta0s, xs):
	theta0s = np.array(theta0s)
	mostSpreadOut = 0
	mostSpreadOutLoc = 0
	variances = []
	for i in range(len(theta0s[0])): #for every columne
		column = theta0s[:, i]
		mean = sum(column)/len(column)
		summ = 0
		for j in column:
			summ += (j-mean)**2
		variance = summ/(len(column)-1)
		variances.append(variance)
		if variance > mostSpreadOut:
			mostSpreadOut = variance
			mostSpreadOutLoc = i
	
	return xs[mostSpreadOutLoc], mostSpreadOut

def getValuesAlong0(title, data,phi_toPlot, highToLow = False, deltas = False):
	
	if highToLow:
		xs = np.concatenate((-1*R[::-1], R)) # high B edge to low B edge
	else:
		xs = R #for center to low B edge 
	
	
	simIO.log.info("The values for {} are given for these radii \n {} \n ".format(title, xs))
	theta0s = []
	
	for i, p in enumerate(phi_toPlot):
		plot_data = np.transpose(data, [2,1,0])[i]
		theta0 = plot_data[-1]# to get 360 degrees which is along the 0 degree
		
		if highToLow:
			# for the high B to center
			middleIndex = (len(plot_data)//2)-1 #This is at 178.988 degrees which is not necessarily parallel to 0 degrees
			theta180 = plot_data[middleIndex][::-1]
			theta0 = np.concatenate((theta180, theta0))
			
		
		theta0s.append(theta0)
		simIO.log.info('{}\n at {}'.format(theta0, p*180/np.pi))
	
	loc, variance = calcMaxDifference(theta0s, xs)
	simIO.log.info("The data is most spread out at {} and the variance is {}".format(loc, variance))

	fig = plt.figure()
	ax = fig.add_subplot()
	if deltas:
		if highToLow:
			plt.title("{} differences based on {}, [High B, Low B]".format(title.capitalize(), PHI[0]*180/np.pi))
		else:
			plt.title("{} differences based on {}, [Center, Low B]".format(title.capitalize(), PHI[0]*180/np.pi))
		for i in range(1, len(theta0s)):
			theta0s[i] = theta0s[i]-theta0s[0]
		theta0s[0] = list(np.zeros(len(theta0s[0])))
		simIO.log.info("Here are the differences \n {} \n ".format(theta0s))
	else:
		if highToLow:
			plt.title("{} magnitudes, [High B, Low B]".format(title.capitalize()))
		else:
			plt.title("{} magnitudes, [Center, Low B]".format(title.capitalize()))
	

	for i, line in enumerate(theta0s):
		ax.plot(xs, line, label = "{:03.1f}".format(PHI[i]*180/np.pi))
	plt.legend(fontsize=4)
	#plt.xticks(np.linspace(0, 0.19, 11))
	plt.xticks(xs[::10])
	plt.tick_params(labelsize=5)
	plt.yticks()
	plt.xlabel("Distance from center - poloidally (m)")
	plt.ylabel("Strength of field (T)")
	plt.grid()
	#plt.show()
	simIO.saveFig(title)
	plt.close()

getValuesAlong0("Norm", Bnorm, PHI, False, True)
getValuesAlong0("Radial", Br, PHI, False, True)
getValuesAlong0("Poloidal", Bpol, PHI, False, True)
getValuesAlong0("Toroidal", Bnorm, PHI, False, True)


## NORM ##
#plot_Xsection('B-field magnitude of HIDRA', Bnorm, 'Bnorm', PHI)

## RADIAL ##
#plot_Xsection('RADIAL B-field magnitude of HIDRA', Br, 'Bradial', PHI)
### POLOIDAL ##
#plot_Xsection('POLOIDAL B-field magnitude of HIDRA', Bpol, 'Bpoloidal', PHI)
### TOROIDAL ##
#plot_Xsection('TOROIDAL B-field magnitude of HIDRA', Btor, 'Btoroidal', PHI)


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