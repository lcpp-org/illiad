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

def getValuesAtDistance(data, phi_toPlot, distOuterWall, R):
	
	radius = round(distOuterWall % 0.19, 4)

	allStrengths = []
	#print(np.degrees(THETA)) # 0 is 2 degs, 5 is 22.11235955, 44 is 178.988, 50 is 203.12359551, and -1 is 360
	for i, p in enumerate(phi_toPlot):
		plot_data = np.transpose(data, [2,1,0])[i]
		#print(plot_data.shape, (len(plot_data)//2)-1)
		
		index = list(R).index(radius)
		strength = []

		for i in (plot_data): #every theta value
			strength.append(i[index])

			
		
		allStrengths.append(strength)	
	
	dic = {}
	for l,phi in enumerate(allStrengths):
		#	print(l)
		nums = []
		for i in phi:
			nums.append(i)
		dic[f"{np.degrees(phi_toPlot[l])}"] = np.array(nums)
	df = pd.DataFrame(dic)
	
	return df
	


def slopes(bigArray, rs):
	bigList = []
	for array in bigArray:
		slopes_list = []
		for i in range(len(rs)-1):
			dr = rs[i]-rs[i+1]
			dB = array[i]-array[i+1]
			slopes_list.append(dB/dr)
		bigList.append(np.array(slopes_list))
	return np.array(bigList)

rampfiles = ['i1q3_hires.npy','i1q3_hires_max.npy','i1q3_hires_noerr_mult.npy','i1q4_hires.npy']

def thetas(radial, poloidal):
	theta_values = np.arange(0, 2*np.pi, np.pi/180)
	theta_bstrengths = ((np.cos(theta_values)*radial)**2 + (np.sin(theta_values)*poloidal)**2)**0.5
	
	slopes_list = []
	for i in range(len(theta_bstrengths)-1):
		dtheta = theta_values[i+1]-theta_values[i]
		dB = theta_bstrengths[i+1]-theta_bstrengths[i]
		slopes_list.append(dB/dtheta)
	
	return np.array(theta_bstrengths), np.array(slopes_list)


def calculateAtDistances(outerwallDist):


	for rampfile in rampfiles[0:1]:
		Bnorm, Br, Bpol, Btor, R, THETA, PHI = loadBs(f'input_files/{rampfile}')
		
		radDic = getValuesAtDistance(Br, PHI, outerwallDist, R)
		polDic = getValuesAtDistance(Bpol, PHI, outerwallDist, R)
		torDic = getValuesAtDistance(Btor, PHI, outerwallDist, R)

		kw15 = np.array([np.array(radDic["18.0"]), np.array(polDic["18.0"]), np.array(torDic["18.0"])])
		spec = np.array([np.array(radDic["126.0"]), np.array(polDic["126.0"]), np.array(torDic["126.0"])])
		nextToPFC = np.array([np.array(radDic["198.0"]), np.array(polDic["198.0"]), np.array(torDic["198.0"])])
		hallprobe = np.array([np.array(radDic["270.0"]), np.array(polDic["270.0"]), np.array(torDic["270.0"])])

		if outerwallDist > 0.19:
			num = 44
		else:
			num = -1

		kw15_thetas, kw15_slopes = thetas(kw15[0][num], kw15[1][num])
		spec_thetas, spec_slopes = thetas(spec[0][num], spec[1][num])
		nextToPFC_thetas, nextToPFC_slopes = thetas(nextToPFC[0][num], nextToPFC[1][num])
		hallprobe_thetas, hallprobe_slopes = thetas(hallprobe[0][num], hallprobe[1][num])


		with open("temp_storage.txt", "a+") as f:
			f.write(f"Filename = \"{rampfile}\" \ndistFromOuterWall = {outerwallDist}\n")
			f.write(f"kw15 = np.array({np.array2string(kw15_thetas, separator=', ')})\n")
			f.write(f"spec = np.array({np.array2string(spec_thetas, separator=', ')})\n")
			f.write(f"nextToPFC = np.array({np.array2string(nextToPFC_thetas, separator=', ')})\n")
			f.write(f"hallprobe = np.array({np.array2string(hallprobe_thetas, separator=', ')})\n")
			f.write(f"kw15_slopes = np.array({np.array2string(kw15_slopes, separator=', ')})\n")
			f.write(f"spec_slopes = np.array({np.array2string(spec_slopes, separator=', ')})\n")
			f.write(f"nextToPFC_slopes = np.array({np.array2string(nextToPFC_slopes, separator=', ')})\n")
			f.write(f"hallprobe_slopes = np.array({np.array2string(hallprobe_slopes, separator=', ')})\n")


def calculateAlong0():

	kw_15_centers = []
	spec_centers = []
	nextToPFC_centers = []
	hallprobe_centers = []

	kw_15_slopes_centers = []
	spec_slopes_centers = []
	nextToPFC_slopes_centers = []
	hallprobe_slopes_centers = []

	for rampfile in rampfiles:
		Bnorm, Br, Bpol, Btor, R, THETA, PHI = loadBs(f'input_files/{rampfile}')
		
		
		radDic = getValuesAlong0(Br, PHI, True)
		polDic = getValuesAlong0(Bpol, PHI, True)
		torDic = getValuesAlong0(Btor, PHI, True)
		
		kw15 = np.array([np.array(radDic["18.0"]), np.array(polDic["18.0"]), np.array(torDic["18.0"])])
		spec = np.array([np.array(radDic["126.0"]), np.array(polDic["126.0"]), np.array(torDic["126.0"])])
		nextToPFC = np.array([np.array(radDic["198.0"]), np.array(polDic["198.0"]), np.array(torDic["198.0"])])
		hallprobe = np.array([np.array(radDic["270.0"]), np.array(polDic["270.0"]), np.array(torDic["270.0"])])
		
		rs = np.linspace(0.38, 0, 39)
		kw15_slopes = slopes(kw15, rs)
		spec_slopes = slopes(spec, rs)
		nextToPFC_slopes = slopes(nextToPFC, rs)
		hallprobe_slopes = slopes(hallprobe, rs)
		
		
		with open("temp_storage.txt", "a+") as f:
			f.write(f"Filename = {rampfile}\n")
			f.write(f"kw15 = np.array({np.array2string(kw15, separator=', ')})\n")
			f.write(f"spec = np.array({np.array2string(spec, separator=', ')})\n")
			f.write(f"nextToPFC = np.array({np.array2string(nextToPFC, separator=', ')})\n")
			f.write(f"hallprobe = np.array({np.array2string(hallprobe, separator=', ')})\n")
			f.write(f"kw15_slopes = np.array({np.array2string(kw15_slopes, separator=', ')})\n")
			f.write(f"spec_slopes = np.array({np.array2string(spec_slopes, separator=', ')})\n")
			f.write(f"nextToPFC_slopes = np.array({np.array2string(nextToPFC_slopes, separator=', ')})\n")
			f.write(f"hallprobe_slopes = np.array({np.array2string(hallprobe_slopes, separator=', ')})\n")

		kw_15_centers.append(np.array([kw15[0][19], kw15[1][19], kw15[2][19]]))
		spec_centers.append(np.array([spec[0][19], spec[1][19], spec[2][19]]))
		nextToPFC_centers.append(np.array([nextToPFC[0][19], nextToPFC[1][19], nextToPFC[2][19]]))
		hallprobe_centers.append(np.array([hallprobe[0][19], hallprobe[1][19], hallprobe[2][19]]))

		kw_15_slopes_centers.append(np.array([kw15_slopes[0][18], kw15_slopes[1][18], kw15_slopes[2][18]]))
		spec_slopes_centers.append(np.array([spec_slopes[0][18], spec_slopes[1][18], spec_slopes[2][18]]))
		nextToPFC_slopes_centers.append(np.array([nextToPFC_slopes[0][18], nextToPFC_slopes[1][18], nextToPFC_slopes[2][18]]))
		hallprobe_slopes_centers.append(np.array([hallprobe_slopes[0][18], hallprobe_slopes[1][18], hallprobe_slopes[2][18]]))

	kw_15_centers = np.array(kw_15_centers)
	spec_centers = np.array(spec_centers)
	nextToPFC_centers = np.array(nextToPFC_centers)
	hallprobe_centers = np.array(hallprobe_centers)

	kw_15_slopes_centers = np.array(kw_15_slopes_centers)
	spec_slopes_centers = np.array(spec_slopes_centers)
	nextToPFC_slopes_centers = np.array(nextToPFC_slopes_centers)
	hallprobe_slopes_centers = np.array(hallprobe_slopes_centers)

	with open("temp_storage.txt", "a+") as f:
		f.write(f"kw15_centers = np.array({np.array2string(kw_15_centers*1000, separator=', ')})\n")
		f.write(f"spec_centers = np.array({np.array2string(spec_centers*1000, separator=', ')})\n")
		f.write(f"nextToPFC_centers = np.array({np.array2string(nextToPFC_centers*1000, separator=', ')})\n")
		f.write(f"hallprobe_centers = np.array({np.array2string(hallprobe_centers*1000, separator=', ')})\n")

		f.write(f"kw15_slopes_centers = np.array({np.array2string(kw_15_slopes_centers*1000, separator=', ')})\n")
		f.write(f"spec_slopes_centers = np.array({np.array2string(spec_slopes_centers*1000, separator=', ')})\n")
		f.write(f"nextToPFC_slopes_centers = np.array({np.array2string(nextToPFC_slopes_centers*1000, separator=', ')})\n")
		f.write(f"hallprobe_slopes_centers = np.array({np.array2string(hallprobe_slopes_centers*1000, separator=', ')})\n")



calculateAtDistances(0.29) #has to be a multiple of 0.01






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