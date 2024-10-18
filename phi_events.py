import numpy as np
import numba as nb
#from math import radians
#from functools import partial

from coordtrans import XYZ_to_RTP
from mesh import *

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def inVV(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return Mesh.a - p_RTP[0]
inVV.direction = -1.0
inVV.terminal = True

"""
def isAngle(t, p_XYZ, Mesh, phi_deg):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi_rad = radians(phi_deg)
	return p_RTP[2] - phi_rad
#isAngle.direction = 1.0


isphi9 = partial(isAngle, phi_deg=9)
isphi9.direction = 1.0

isphi18 = partial(isAngle, phi_deg=18.)
isphi18.direction = 1.0

isphi27 = partial(isAngle, phi_deg=27)
isphi27.direction = 1.0

isphi36 = partial(isAngle, phi_deg=36.)
isphi36.direction = 1.0

isphi45 = partial(isAngle, phi_deg=45.)
isphi45.direction = 1.0

isphi54 = partial(isAngle, phi_deg=54.)
isphi54.direction = 1.0

isphi63 = partial(isAngle, phi_deg=63.)
isphi63.direction = 1.0

isphi72 = partial(isAngle, phi_deg=72.)
isphi72.direction = 1.0

isphi81 = partial(isAngle, phi_deg=81.)
isphi81.direction = 1.0

isphi90 = partial(isAngle, phi_deg=90.)
isphi90.direction = 1.0

isphi99 = partial(isAngle, phi_deg=99.)
isphi99.direction = 1.0

isphi108 = partial(isAngle, phi_deg=108.)
isphi108.direction = 1.0

isphi117 = partial(isAngle, phi_deg=117.)
isphi117.direction = 1.0

isphi126 = partial(isAngle, phi_deg=126.)
isphi126.direction = 1.0

isphi135 = partial(isAngle, phi_deg=135.)
isphi135.direction = 1.0

isphi144 = partial(isAngle, phi_deg=144.)
isphi144.direction = 1.0

isphi153 = partial(isAngle, phi_deg=153.)
isphi153.direction = 1.0

isphi162 = partial(isAngle, phi_deg=162.)
isphi162.direction = 1.0

isphi171 = partial(isAngle, phi_deg=171.)
isphi171.direction = 1.0

isphi180 = partial(isAngle, phi_deg=180.)
isphi180.direction = 1.0

isphi189 = partial(isAngle, phi_deg=189.)
isphi189.direction = 1.0

isphi198 = partial(isAngle, phi_deg=198.)
isphi198.direction = 1.0

isphi207 = partial(isAngle, phi_deg=207.)
isphi207.direction = 1.0

isphi216 = partial(isAngle, phi_deg=216.)
isphi216.direction = 1.0

isphi225 = partial(isAngle, phi_deg=225.)
isphi225.direction = 1.0

isphi234 = partial(isAngle, phi_deg=234.)
isphi234.direction = 1.0

isphi243 = partial(isAngle, phi_deg=243.)
isphi243.direction = 1.0

isphi252 = partial(isAngle, phi_deg=252.)
isphi252.direction = 1.0

isphi261 = partial(isAngle, phi_deg=261.)
isphi261.direction = 1.0

isphi270 = partial(isAngle, phi_deg=270.)
isphi270.direction = 1.0

isphi279 = partial(isAngle, phi_deg=279.)
isphi279.direction = 1.0

isphi288 = partial(isAngle, phi_deg=288.)
isphi288.direction = 1.0

isphi297 = partial(isAngle, phi_deg=297.)
isphi297.direction = 1.0

isphi306 = partial(isAngle, phi_deg=306.)
isphi306.direction = 1.0

isphi315 = partial(isAngle, phi_deg=315.)
isphi315.direction = 1.0

isphi324 = partial(isAngle, phi_deg=324.)
isphi324.direction = 1.0

isphi333 = partial(isAngle, phi_deg=333.)
isphi333.direction = 1.0

isphi342 = partial(isAngle, phi_deg=342.)
isphi342.direction = 1.0

isphi351 = partial(isAngle, phi_deg=351.)
isphi351.direction = 1.0


def make_event(phi_deg):

	def _function(t, p_XYZ, Mesh):
		#return partial(isAngle, phi_deg=9.)
		p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
		phi_rad = radians(phi_deg)
		return p_RTP[2] - phi_rad
	
	_function.direction = 1.0
	_function.__name__ = 'isphi'+ ('{:.0f}'.format(phi_deg))
	print('Generated event function: {}'.format(_function.__name__))

	return _function

"""

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi9(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 1. * (np.pi/20.) 
isphi9.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi18(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 2. * (np.pi/20.)
isphi18.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi27(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 3. * (np.pi/20.)
isphi27.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi36(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 4. * (np.pi/20.)
isphi36.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi45(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 5. * (np.pi/20.)
isphi45.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi54(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 6. * (np.pi/20.)
isphi54.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi63(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 7. * (np.pi/20.) 
isphi63.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi72(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 8. * (np.pi/20.)
isphi72.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi81(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 9. * (np.pi/20.)
isphi81.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi90(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 10. * (np.pi/20.)
isphi90.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi99(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 11. * (np.pi/20.)
isphi99.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi108(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 12. * (np.pi/20.)
isphi108.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi117(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 13. * (np.pi/20.)
isphi117.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi126(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 14. * (np.pi/20.)
isphi126.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi135(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 15. * (np.pi/20.)
isphi135.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi144(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 16. * (np.pi/20.)
isphi144.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi153(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 17. * (np.pi/20.)
isphi153.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi162(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 18. * (np.pi/20.)
isphi162.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi171(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 19. * (np.pi/20.)
isphi171.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi180(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 20. * (np.pi/20.)
isphi180.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi189(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 21. * (np.pi/20.)
isphi189.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi198(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 22. * (np.pi/20.)
isphi198.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi207(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 23. * (np.pi/20.)
isphi207.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi216(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 24. * (np.pi/20.)
isphi216.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi225(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 25. * (np.pi/20.)
isphi225.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi234(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 26. * (np.pi/20.)
isphi234.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi243(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 27. * (np.pi/20.)
isphi243.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi252(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 28. * (np.pi/20.)
isphi252.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi261(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 29. * (np.pi/20.)
isphi261.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi270(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 30. * (np.pi/20.)
isphi270.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi279(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 31. * (np.pi/20.)
isphi279.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi288(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 32. * (np.pi/20.)
isphi288.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi297(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 33. * (np.pi/20.)
isphi297.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi306(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 34. * (np.pi/20.)
isphi306.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi315(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 35. * (np.pi/20.)
isphi315.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi324(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 36. * (np.pi/20.)
isphi324.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi333(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 37. * (np.pi/20.)
isphi333.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi342(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 38. * (np.pi/20.)
isphi342.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi351(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 39. * (np.pi/20.)
isphi351.direction = 1.0


#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi360(t, p_XYZ, Mesh):
	return p_XYZ[1]
isphi360.direction = -1


#poincare_events = [ inVV, 
#                    isphi9, 
#                    isphi18, 
#                    isphi27, 
#                    isphi36, 
#                    isphi45, 
#                    isphi54, 
#                    isphi63, 
#                    isphi72,
#                    isphi81, 
#                    isphi90, 
#                    isphi99, 
#                    isphi108, 
#                    isphi117, 
#                    isphi126, 
#                    isphi135, 
#                    isphi144,
#                    isphi153, 
#                    isphi162, 
#                    isphi171, 
#                    isphi180, 
#                    isphi189,
#                    isphi198,
#                    isphi207, 
#                    isphi216, 
#                    isphi225, 
#                    isphi234, 
#                    isphi243,
#                    isphi252, 
#                    isphi261, 
#                    isphi270, 
#                    isphi279, 
#                    isphi288,
#                    isphi297, 
#                    isphi306, 
#                    isphi315,
#                    isphi324, 
#                    isphi333,
#                    isphi342, 
#                    isphi351,
#                    isphi360
#                ]
