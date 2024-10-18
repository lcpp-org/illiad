import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, colors, cm, colormaps
import copy

import class_outputHandler as out
from mesh import *
from coordtrans import *



## SET UP RUN DIRECTORY
simIO = out.IOHandler("HIDRA_1q3ERR_2000s") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
#simIO = out.IOHandler("HIDRA_1q4ERR_1500s") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simIO.startLog()

## DEFINE MESH AND LOAD FIELD
BX, BY, BZ = np.load('input_files/HIDRA_i3ERR_hires.npy')
#BX, BY, BZ = np.load('input_files/HIDRA_i4ERR_hires.npy')
mesh_prd = np.array([0, 1, 5], dtype=np.int32)
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(BX, BY, BZ, mesh_prd, errField=True)

## Port Plotting
def plotPorts(figure):
    ax_ = figure.add_subplot()

    # Import data on HIDRA port size/locations for plotting
    ports = simIO.loadPorts_fromCSV('input_files/HIDRA_ports.csv')
    for port in ports.T:
        port_plot = patches.Ellipse((port[0], port[1]), port[2], port[3],
                                    fill=True, alpha=0.2, facecolor='black', edgecolor='black', linewidth=0.0)
        ax_.add_patch(port_plot)


## GET DATA
wallPtArray = simIO.loadNumpyData('Wallpoints_0-10-20mm.npy')
dr_String = '0-10-20'
ion_temp_eV = 2. #eV

#print(f'{wallPtArray=}')
print('b_hidra.dr:{}'.format(b_hidra.dr))
print('b_hidra.dtheta:{dth}'.format(dth=b_hidra.dtheta*180/np.pi)) 
print('b_hidra.dphi:{dph}'.format(dph=b_hidra.dphi*180/np.pi) )
print('b_hidra.nr:{}'.format(b_hidra.nr))
print('b_hidra.ntheta:{nth}'.format(nth=b_hidra.ntheta)) 
print('b_hidra.nphi:{nph}'.format(nph=b_hidra.nphi) )

##################################
## POST-SOLVER OUTPUT (WALL PLOT)
simIO.log.info('Plotting wall hits, total events = {}...'.format(wallPtArray[0].size))


phi_plot = wallPtArray[2]*(-1) + 2*np.pi
theta_plot = wallPtArray[1]
for i in range(len(theta_plot)):
    if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi

phi_plot_deg = (phi_plot*(180/np.pi) + 180. + 0.) % 360.
theta_plot_deg = theta_plot*(180/np.pi)

## CREATE HISTOGRAM
phi_edges = np.linspace(0, 360, 361)
theta_edges = np.linspace(-180, 180, 181)
H, phi_edges, theta_edges = np.histogram2d(phi_plot_deg, theta_plot_deg, bins=[phi_edges, theta_edges], density=True)
H = H.T # histogram reverse axes for some reason; transpose:


## PLOT HISTOGRAM
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'figure.autolayout':True})
fig3 = plt.figure()
ax3 = fig3.add_subplot()
plotPorts(fig3)

############################################################################################################
plt.imshow(H, interpolation='nearest', origin='lower',
            extent=[phi_edges[0], phi_edges[-1], theta_edges[0], theta_edges[-1]],
            aspect=0.2, cmap='Blues', norm=colors.LogNorm(vmin=1E-5, vmax=1E-3))
plt.colorbar(location='bottom', shrink=0.7)
plt.grid(which='both', linewidth=0.25)

plt.xlabel('Toroidal Angle, $\phi$, $[\degree]$')
plt.xlim(0, 360)
plt.xticks(np.linspace(9, 360, 40))
ax3.xaxis.set_tick_params(labelsize=3.5)

plt.ylabel('Poloidal Location')
plt.ylim(-180, 180)
plt.yticks(np.linspace(-180, 180, 5), ['Inner   \nMidplane', 'Bottom', 'Outer   \nMidplane', 'Top', 'Inner   \nMidplane'])
ax3.yaxis.set_tick_params(labelsize=5)

simIO.saveFig('Wall_Histogram_{}mm_{}eV.png'.format(dr_String, int(ion_temp_eV)) )
plt.close()



## Plot Discrete Wall Points
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'figure.autolayout':True})
fig = plt.figure()
ax = fig.add_subplot(polar=False, aspect=0.2)
plotPorts(fig)

# plot wall event locations
plt.scatter(phi_plot_deg, theta_plot_deg, s=0.25, c='k', linewidths=0.0)
ax.grid(linewidth = 0.25, linestyle=':', c='grey')

plt.xlabel('Toroidal Angle, $\phi$, $[\degree]$')
plt.xlim(0, 360)
plt.xticks(np.linspace(9, 360, 40))
ax.xaxis.set_tick_params(labelsize=3.5)

plt.ylabel('Poloidal Location')
plt.ylim(-180, 180)
plt.yticks(np.linspace(-180, 180, 5), ['Inner Midplane', 'Bottom', 'Outer Midplane', 'Top', 'Inner Midplane'])
ax.yaxis.set_tick_params(labelsize=5)

plt.title('Distribution of Field Line Intersections with HIDRA Wall\n'
          +'Particle: $Li^+, T={}eV$'.format(ion_temp_eV))

simIO.saveFig('Wallpoints_BorisPts_{}mm_{}eV.png'.format(dr_String, int(ion_temp_eV)) )
plt.close()
simIO.log.info('...finished\n')



########################################
## POST-SOLVER OUTPUT ( *3D* WALL PLOT)
########################################
simIO.log.info('Attempting 3D plot...')

rad_plot = wallPtArray[0]
theta_plot = wallPtArray[1]
phi_plot = wallPtArray[2] + 2*np.pi

simIO.log.info('r{}, th{}, ph{}'.format(len(rad_plot), len(theta_plot), len(phi_plot)))

xyz_plt = np.zeros(shape=(len(theta_plot), 3))
for i in range(len(theta_plot)):
    if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi
    xyz_plt[i] = RTP_to_XYZ( np.array([rad_plot[i], theta_plot[i], phi_plot[i]]), b_hidra.R0 )



fig = plt.figure()
ax2 = fig.add_subplot(projection='3d', computed_zorder=True)

## PLOT VACUUM VESSEL TORUS
#############################
#ptheta = np.linspace(0, 2*np.pi, 181)
ptheta = np.linspace(-np.pi, np.pi, 181)
pphi   = np.linspace(0, 2.*np.pi, 181)
ptheta, pphi = np.meshgrid(ptheta, pphi)

px = (b_hidra.R0 + b_hidra.a*np.cos(ptheta)) * np.cos(pphi)
py = (b_hidra.R0 + b_hidra.a*np.cos(ptheta)) * np.sin(pphi)
pz = b_hidra.a * np.sin(ptheta)


## CREATE HISTOGRAM 2
phi_edges = np.linspace(0, 360, 181)
theta_edges = np.linspace(-180, 180, 181)
H_2, phi_edges, theta_edges = np.histogram2d(phi_plot_deg, theta_plot_deg, bins=[phi_edges, theta_edges], density=False)


## Set up histogram output as colormap data ############################
color_dimension = H_2
#print(f'{H_2.min()=}')
#print(f'{H_2.max()=}')
minn= 1E-8
maxx=  1E-3

#norm = colors.Normalize(vmin=1E-5, vmax=6E-4)
norm = colors.LogNorm()#vmin=3E-4, vmax=7E-4)
#print(f'{norm.vmin=}')
#print(f'{norm.vmax=}')
my_cmap = copy.copy(colormaps['Blues'])
my_cmap.set_bad(my_cmap(0))

m = plt.cm.ScalarMappable(norm=norm, cmap=my_cmap)
m.set_array([])
fcolors = m.to_rgba(color_dimension)

ax2.plot_surface(px, py, pz, rstride=1, cstride=1,
                 vmin=minn, vmax=maxx,
                 facecolors=fcolors,
                 edgecolor='grey', linewidth=0.1, 
                 alpha=1.0, shade=False) #, zorder=1)

ax2.set_xlim3d(-1, 1)
ax2.set_ylim3d(-1, 1)
ax2.set_zlim3d(-0.7, 0.7)
ax2._axis3don = False
ax2.elev -= 12
ax2.azim += 10

plt.title('Distribution of Field Line Intersections with HIDRA Wall\n'
          +'Particle: $Li^+, T={}eV$'.format(ion_temp_eV))

simIO.saveFig('Wallpoints3D_BorisPts_{}mm_{}eV.png'.format(dr_String, int(ion_temp_eV)))
plt.close()
simIO.log.info('...finished\n')