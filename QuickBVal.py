## IMPORT
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


def loadBs(filename):
    ## DEFINE MESH AND LOAD FIELD
    Bx, By, Bz = np.load(filename)
    mesh_prd = np.array([0, 1, 5], dtype=np.int32)
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(Bx, By, Bz, mesh_prd, errField=True)


    mesh_ntheta = int(b_hidra.ntheta/2)
    mesh_dtheta = b_hidra.dtheta*2
	
    R     = np.linspace( b_hidra.r_min,       b_hidra.r_max,    int((b_hidra.nr//2)+1))
    THETA = np.linspace( b_hidra.theta_min, b_hidra.theta_max, mesh_ntheta)
    #PHI   = np.linspace( b_hidra.phi_min,     b_hidra.phi_max,   int(b_hidra.nphi/2))
    PHI   = np.array([18,45, 54, 90, 117, 126, 162, 189, 198, 234,261, 270, 306, 333, 342])*(np.pi/180)#np.linspace( 9*(np.pi/180),     2*np.pi,   40)

   
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
    return Bnorm, Br, Bpol, Btor, R, THETA, PHI




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

def getValuesAlong0(data,phi_toPlot, highToLow = False):

	#simIO.log.info("The values for {} are given for these radii \n {} \n ".format(title, xs))
	theta0s = []
	#print(np.degrees(THETA)) # 0 is 2 degs, 5 is 22.11235955, 44 is 178.988, 50 is 203.12359551, and -1 is 360
	for i, p in enumerate(phi_toPlot):
		plot_data = np.transpose(data, [2,1,0])[i]
		#print(plot_data.shape, (len(plot_data)//2)-1)
		'''if np.degrees(p)%72 == 45:
			theta0 = plot_data[5]
			middleIndex = 50
		else:'''
		theta0 = plot_data[-1]
		middleIndex = 44
		
		if highToLow:
			# for the high B to center
			theta180 = plot_data[middleIndex][::-1]
			theta0 = np.concatenate((theta180, theta0))
			
		
		theta0s.append(theta0)
		#simIO.log.info('{}\n at {}'.format(theta0, p*180/np.pi))
	
	dic = {}
	for l,phi in enumerate(theta0s):
		#	print(l)
		nums = []
		for i in range(0,96,5):
			nums.append(phi[i])
		for k in range(101,192,5):
			nums.append(phi[k])
		dic[f"{np.degrees(phi_toPlot[l])}"] = np.array(nums)
	df = pd.DataFrame(dic)
	
	return df

def getValuesAtDistance(title, data, phi_toPlot):
	
	
	print("The values for {} are given for these theta values \n {} \n ".format(title, np.degrees(THETA)))
	allStrengths = []
	#print(np.degrees(THETA)) # 0 is 2 degs, 5 is 22.11235955, 44 is 178.988, 50 is 203.12359551, and -1 is 360
	for i, p in enumerate(phi_toPlot):
		plot_data = np.transpose(data, [2,1,0])[i]
		#print(plot_data.shape, (len(plot_data)//2)-1)
		
		radius = 0.1 #distance from the poloidal center outward
		index = list(R).index(radius)
		strength = []

		for i in (plot_data): #every theta value
			strength.append(i[index])

			
		
		allStrengths.append(strength)
		print('{}\n at {}'.format(allStrengths, p*180/np.pi))

	fig = plt.figure()
	ax = fig.add_subplot()
	
	plt.title("{} strengths at radius of {}".format(title.capitalize(), -1*(radius-0.19)))#uses radius to give distance inward from outer wall
	
	rangeof23 = []
	
	for i, line in enumerate(allStrengths):
		ax.plot(np.degrees(THETA), line, label = "{:03.1f}".format(PHI[i]*180/np.pi))
		
		if PHI[i]*180/np.pi%72 == 18:
			rangeof23.append(line[-1]+0.0023)
	
	ymin,ymax = ax.get_ylim()
	vline_positions = np.array([360, 22.5, 45, 90, 180])
	ax.vlines(vline_positions, ymin=ymin, ymax=ymax,linestyles="dashed")
	for x_pos, label in zip(vline_positions, ["360 (0)", "22.5", "45", "90", "180"]):
		ax.text(x_pos, ymax, label, horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='black')

	ax.hlines(np.array([min(rangeof23), max(rangeof23)]), 0, 360, colors="black", linestyles="dashed", label = "23 G larger than the $\theta$=0 angle at ")
	for y_pos, label in zip(np.array([min(rangeof23), max(rangeof23)]), [f"{min(rangeof23):0.3f}", f"{max(rangeof23):0.3f}"]):
		ax.text(360, y_pos, label, horizontalalignment='left', verticalalignment='center', fontsize=10, color='black')
	plt.legend()
	#plt.xticks(np.linspace(0, 0.19, 11))
	ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  # Major ticks every 2 units
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # Minor ticks every 0.5 units
	ax.yaxis.set_major_locator(ticker.MultipleLocator(0.005))  # Major y-ticks every 0.5
	ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.001))
	
	plt.yticks()
	
	plt.ylabel("Strength of field (T)")
	plt.grid()
	plt.show()
	#simIO.saveFig("" + title + "at" + radius)
	plt.close()


rampfiles = ['i1q3_hires_500t.npy','i1q3_hires_1000t.npy','i1q3_hires_1500t.npy','i1q3_hires_2000t.npy','i1q3_hires_2500t.npy', 'i1q3_hires_3000t.npy','i1q3_hires_3300t.npy']
for rampfile in rampfiles:
    Bnorm, Br, Bpol, Btor, R, THETA, PHI = loadBs(f'input_files/{rampfile}')
    
    radDic = getValuesAlong0(Br, PHI, True)
    polDic = getValuesAlong0(Bpol, PHI, True)
    torDic = getValuesAlong0(Btor, PHI, True)
	
    kw15 = np.array(tempDic["18.0"])
    spec = np.array(tempDic["126.0"])
    nextToPFC = np.array(tempDic["198.0"])
    hallprobe = np.array(tempDic["270.0"])


    def slopes(array):
        rs = np.linspace(0.38, 0, 39)
        slopes_list = []
        for i in range(len(rs)-1):
            dr = rs[i]-rs[i+1]
            dB = array[i]-array[i+1]
            slopes_list.append(dB/dr)
        slopes_arr = np.flip(np.array(slopes_list)) #From 0 to 0.38 slopes
        #print()
        return slopes_arr

    kw15_slopes = slopes(kw15)
    spec_slopes = slopes(spec)
    nextToPFC_slopes = slopes(nextToPFC)
    hallprobe_slopes = slopes(hallprobe)



    with open("temp_storage.txt", "a+") as f:
        f.write("Poloidal, 1q3_hires\n")
        f.write(f"kw15 = {kw15}\n")
        f.write(f"spec = {spec}\n")
        f.write(f"nextToPFC = {nextToPFC}\n")
        f.write(f"hallprobe = {hallprobe}\n")
        f.write(f"kw15_slopes = {kw15_slopes}\n")
        f.write(f"spec_slopes = {spec_slopes}\n")
        f.write(f"nextToPFC_slopes = {nextToPFC_slopes}\n")
        f.write(f"hallprobe_slopes = {hallprobe_slopes}\n")






    #print(f"15kW - .29 : {kw15[9]*10000}, .19 : {kw15[19]*10000}, .09 : {kw15[29]*10000}")
    #print(f"Spec - .29 : {spec[9]*10000}, .19 : {spec[19]*10000}, .09 : {spec[29]*10000}")
    #print(f"Next To PFC - .29 : {nextToPFC[9]*10000}, .19 : {nextToPFC[19]*10000}, .09 : {nextToPFC[29]*10000}")
    #print(f"Hall Probe - .29 : {hallprobe[9]*10000}, .19 : {hallprobe[19]*10000}, .09 : {hallprobe[29]*10000}")
    #print(f"np.array({[[kw15[9]*10000, kw15[19]*10000, kw15[29]*10000], [spec[9]*10000, spec[19]*10000, spec[29]*10000], [nextToPFC[9]*10000, nextToPFC[19]*10000, nextToPFC[29]*10000], [hallprobe[9]*10000, hallprobe[19]*10000, hallprobe[29]*10000]]})")


    ## NORM ##
    #plot_Xsection('B-field magnitude of HIDRA', Bnorm, 'Bnorm', PHI)

    ## RADIAL ##
    #plot_Xsection('RADIAL B-field magnitude of HIDRA', Br, 'Bradial', PHI)
    ### POLOIDAL ##
    #plot_Xsection('POLOIDAL B-field magnitude of HIDRA', Bpol, 'Bpoloidal', PHI)
    ### TOROIDAL ##
    #plot_Xsection('TOROIDAL B-field magnitude of HIDRA', Btor, 'Btoroidal', PHI)