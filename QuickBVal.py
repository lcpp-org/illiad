## IMPORT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mesh import *


def loadBs(filename):
	'''
	Function to generate the B field dictionaries and returns them and the three arrays used to mesh the vessel
	PHI controls the phi angles at which the B field is tabulated, the key for the dictionary is the phi angle and each angle has its
	respective array of values for all of the radii and theta at that phi angle

	Bnorm, Br, Bpol, Btor, R, THETA, PHI
	'''
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


def getValuesAlong0(data,phi_toPlot, highToLow = False):

	'''
	Gets the values along theta equals 0 for the phi angle in question and returns them as a dictionary
	'''

	theta0s = []
	
	for i, p in enumerate(phi_toPlot):
		plot_data = np.transpose(data, [2,1,0])[i]
		
		#print(plot_data.shape, (len(plot_data)//2)-1)
		'''if np.degrees(p)%72 == 45:
			theta0 = plot_data[5]
			middleIndex = 50
		else:'''
		
		theta0 = plot_data[-1]

		#print(np.degrees(THETA)) # 0 is 2 degs, 5 is 22.11235955, 44 is 178.988, 50 is 203.12359551, and -1 is 360

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
	'''
	Gets the values at a certain distance from the poloidal center (radius) for every theta value available
	'''
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
	'''
	Calculates the slope by taking two values and then dividing them by the difference between their locations
	
	the returned array has smaller length than the original array
	'''

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

def thetas(magField1, magField2):
	'''
	Function to calculate the error in the plane given by the two magnetic fields
	'''
	errorAngle = 10
	delta = np.radians(np.linspace(-errorAngle, errorAngle, (2*errorAngle)+1))


	deltaB1 = magField1*(1-np.cos(delta)) + magField2*(np.sin(delta))
	deltaB2 = magField1*(np.sin(delta)) + magField2*(1-np.cos(delta))

	
	return delta, np.array([deltaB1, deltaB2])


def dBdtheta(outerwallDist, directions):
	'''
	This function calculates the change in magnetic field  at different angles by loading the magnetic field for that distance,
	getting the two magnetic fields in the plane of rotation required and then using the thetas function to get the change in strength,
	while changing the theta. 
	'''
	magFieldDir1, magFieldDir2 = directions

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

		kw15_thetas, kw15_slopes = thetas(kw15[magFieldDir1][num], kw15[magFieldDir2][num])
		spec_thetas, spec_slopes = thetas(spec[magFieldDir1][num], spec[magFieldDir2][num])
		nextToPFC_thetas, nextToPFC_slopes = thetas(nextToPFC[magFieldDir1][num], nextToPFC[magFieldDir2][num])
		hallprobe_thetas, hallprobe_slopes = thetas(hallprobe[magFieldDir1][num], hallprobe[magFieldDir2][num])

		dirNames = ["Radial", "Poloidal", "Toroidal"]

		with open("temp_storage.txt", "a+") as f:
			f.write(f"Filename = \"{rampfile}\" \ndistFromOuterWall = {outerwallDist}\ndir1 = \"{dirNames[magFieldDir1]}\"\ndir2 = \"{dirNames[magFieldDir2]}\"\n")
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



dBdtheta(0.09, [0,1]) #has to be a multiple of 0.01






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