import os
import sys
import numpy as np
import glob

Rmaj = 0.72 #[m]
Rmin = 0.19 #[m]

path = '500spins_24_n015_p100_DOP853_1E-3_1E-12_1E-9/'
dir ='indLines'

try:
	os.mkdir(path+dir)
except OSError as error:
	print(error)

phi_plot = 36.
# import files at phi=36deg.
files = sorted(glob.glob(path + 'Poincare_output_4_*.npy'))

##################
## PLOTTING SETUP
##################
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.style as mplstyle

#mplstyle.use(['dark_background', 'fast'])
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

for f in files:
	print(f)
	fig = plt.figure()
	ax = fig.add_subplot(111, polar=True)
	
	th_f, r_f = np.load(f)
	#plt.scatter(th_f, r_f, s=0.1, c=UIUCcol[int(np.fmod(i,len(UIUCcol))
	plt.scatter(th_f, r_f, s=0.1)
	
	ax.set_rmax(Rmin)
	
	flabels = f.split('_')
	runNum = flabels[-1][:-4]
	plt.title(r'Poincare Plot, run #'+runNum)
	plt.savefig(path+'indLines/Poincare_'+runNum+'.png', dpi=900)
	#plt.show()
	plt.close()

"""#################
## POINCARE PLOTS
#################
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
	
#	for i in range(1, np.size(Poincare_output)):
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
#plt.show()
plt.close('all')
"""