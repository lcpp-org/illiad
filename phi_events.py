import numpy as np
#import numba as nb
from coordtrans import XYZ_to_RTP
from mesh import *
import phi_events 
#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def inVV(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return Mesh.a - p_RTP[0]
inVV.direction = -1.0
inVV.terminal = True




#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi1(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 1. * (np.pi/180)
	return phi - 1. * (np.pi/180)
isphi1.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi2(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 2. * (np.pi/180)
	return phi - 2. * (np.pi/180)
isphi2.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi3(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 3. * (np.pi/180)
	return phi - 3. * (np.pi/180)
isphi3.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi4(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 4. * (np.pi/180)
	return phi - 4. * (np.pi/180)
isphi4.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi5(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 5. * (np.pi/180)
	return phi - 5. * (np.pi/180)
isphi5.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi6(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 6. * (np.pi/180)
	return phi - 6. * (np.pi/180)
isphi6.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi7(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 7. * (np.pi/180)
	return phi - 7. * (np.pi/180)
isphi7.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi8(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 8. * (np.pi/180)
	return phi - 8. * (np.pi/180)
isphi8.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi9(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 9. * (np.pi/180)
	return phi - 9. * (np.pi/180)
isphi9.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi10(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 10. * (np.pi/180)
	return phi - 10. * (np.pi/180)
isphi10.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi11(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 11. * (np.pi/180)
	return phi - 11. * (np.pi/180)
isphi11.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi12(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 12. * (np.pi/180)
	return phi - 12. * (np.pi/180)
isphi12.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi13(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 13. * (np.pi/180)
	return phi - 13. * (np.pi/180)
isphi13.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi14(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 14. * (np.pi/180)
	return phi - 14. * (np.pi/180)
isphi14.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi15(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 15. * (np.pi/180)
	return phi - 15. * (np.pi/180)
isphi15.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi16(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 16. * (np.pi/180)
	return phi - 16. * (np.pi/180)
isphi16.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi17(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 17. * (np.pi/180)
	return phi - 17. * (np.pi/180)
isphi17.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi18(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 18. * (np.pi/180)
	return phi - 18. * (np.pi/180)
isphi18.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi19(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 19. * (np.pi/180)
	return phi - 19. * (np.pi/180)
isphi19.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi20(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 20. * (np.pi/180)
	return phi - 20. * (np.pi/180)
isphi20.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi21(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 21. * (np.pi/180)
	return phi - 21. * (np.pi/180)
isphi21.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi22(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 22. * (np.pi/180)
	return phi - 22. * (np.pi/180)
isphi22.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi23(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 23. * (np.pi/180)
	return phi - 23. * (np.pi/180)
isphi23.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi24(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 24. * (np.pi/180)
	return phi - 24. * (np.pi/180)
isphi24.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi25(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 25. * (np.pi/180)
	return phi - 25. * (np.pi/180)
isphi25.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi26(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 26. * (np.pi/180)
	return phi - 26. * (np.pi/180)
isphi26.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi27(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 27. * (np.pi/180)
	return phi - 27. * (np.pi/180)
isphi27.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi28(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 28. * (np.pi/180)
	return phi - 28. * (np.pi/180)
isphi28.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi29(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 29. * (np.pi/180)
	return phi - 29. * (np.pi/180)
isphi29.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi30(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 30. * (np.pi/180)
	return phi - 30. * (np.pi/180)
isphi30.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi31(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 31. * (np.pi/180)
	return phi - 31. * (np.pi/180)
isphi31.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi32(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 32. * (np.pi/180)
	return phi - 32. * (np.pi/180)
isphi32.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi33(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 33. * (np.pi/180)
	return phi - 33. * (np.pi/180)
isphi33.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi34(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 34. * (np.pi/180)
	return phi - 34. * (np.pi/180)
isphi34.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi35(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 35. * (np.pi/180)
	return phi - 35. * (np.pi/180)
isphi35.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi36(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 36. * (np.pi/180)
	return phi - 36. * (np.pi/180)
isphi36.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi37(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 37. * (np.pi/180)
	return phi - 37. * (np.pi/180)
isphi37.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi38(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 38. * (np.pi/180)
	return phi - 38. * (np.pi/180)
isphi38.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi39(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 39. * (np.pi/180)
	return phi - 39. * (np.pi/180)
isphi39.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi40(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 40. * (np.pi/180)
	return phi - 40. * (np.pi/180)
isphi40.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi41(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 41. * (np.pi/180)
	return phi - 41. * (np.pi/180)
isphi41.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi42(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 42. * (np.pi/180)
	return phi - 42. * (np.pi/180)
isphi42.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi43(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 43. * (np.pi/180)
	return phi - 43. * (np.pi/180)
isphi43.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi44(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 44. * (np.pi/180)
	return phi - 44. * (np.pi/180)
isphi44.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi45(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 45. * (np.pi/180)
	return phi - 45. * (np.pi/180)
isphi45.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi46(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 46. * (np.pi/180)
	return phi - 46. * (np.pi/180)
isphi46.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi47(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 47. * (np.pi/180)
	return phi - 47. * (np.pi/180)
isphi47.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi48(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 48. * (np.pi/180)
	return phi - 48. * (np.pi/180)
isphi48.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi49(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 49. * (np.pi/180)
	return phi - 49. * (np.pi/180)
isphi49.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi50(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 50. * (np.pi/180)
	return phi - 50. * (np.pi/180)
isphi50.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi51(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 51. * (np.pi/180)
	return phi - 51. * (np.pi/180)
isphi51.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi52(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 52. * (np.pi/180)
	return phi - 52. * (np.pi/180)
isphi52.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi53(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 53. * (np.pi/180)
	return phi - 53. * (np.pi/180)
isphi53.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi54(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 54. * (np.pi/180)
	return phi - 54. * (np.pi/180)
isphi54.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi55(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 55. * (np.pi/180)
	return phi - 55. * (np.pi/180)
isphi55.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi56(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 56. * (np.pi/180)
	return phi - 56. * (np.pi/180)
isphi56.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi57(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 57. * (np.pi/180)
	return phi - 57. * (np.pi/180)
isphi57.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi58(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 58. * (np.pi/180)
	return phi - 58. * (np.pi/180)
isphi58.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi59(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 59. * (np.pi/180)
	return phi - 59. * (np.pi/180)
isphi59.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi60(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 60. * (np.pi/180)
	return phi - 60. * (np.pi/180)
isphi60.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi61(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 61. * (np.pi/180)
	return phi - 61. * (np.pi/180)
isphi61.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi62(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 62. * (np.pi/180)
	return phi - 62. * (np.pi/180)
isphi62.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi63(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 63. * (np.pi/180)
	return phi - 63. * (np.pi/180)
isphi63.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi64(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 64. * (np.pi/180)
	return phi - 64. * (np.pi/180)
isphi64.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi65(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 65. * (np.pi/180)
	return phi - 65. * (np.pi/180)
isphi65.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi66(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 66. * (np.pi/180)
	return phi - 66. * (np.pi/180)
isphi66.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi67(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 67. * (np.pi/180)
	return phi - 67. * (np.pi/180)
isphi67.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi68(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 68. * (np.pi/180)
	return phi - 68. * (np.pi/180)
isphi68.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi69(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 69. * (np.pi/180)
	return phi - 69. * (np.pi/180)
isphi69.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi70(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 70. * (np.pi/180)
	return phi - 70. * (np.pi/180)
isphi70.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi71(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 71. * (np.pi/180)
	return phi - 71. * (np.pi/180)
isphi71.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi72(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 72. * (np.pi/180)
	return phi - 72. * (np.pi/180)
isphi72.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi73(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 73. * (np.pi/180)
	return phi - 73. * (np.pi/180)
isphi73.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi74(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 74. * (np.pi/180)
	return phi - 74. * (np.pi/180)
isphi74.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi75(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 75. * (np.pi/180)
	return phi - 75. * (np.pi/180)
isphi75.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi76(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 76. * (np.pi/180)
	return phi - 76. * (np.pi/180)
isphi76.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi77(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 77. * (np.pi/180)
	return phi - 77. * (np.pi/180)
isphi77.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi78(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 78. * (np.pi/180)
	return phi - 78. * (np.pi/180)
isphi78.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi79(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 79. * (np.pi/180)
	return phi - 79. * (np.pi/180)
isphi79.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi80(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 80. * (np.pi/180)
	return phi - 80. * (np.pi/180)
isphi80.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi81(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 81. * (np.pi/180)
	return phi - 81. * (np.pi/180)
isphi81.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi82(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 82. * (np.pi/180)
	return phi - 82. * (np.pi/180)
isphi82.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi83(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 83. * (np.pi/180)
	return phi - 83. * (np.pi/180)
isphi83.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi84(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 84. * (np.pi/180)
	return phi - 84. * (np.pi/180)
isphi84.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi85(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 85. * (np.pi/180)
	return phi - 85. * (np.pi/180)
isphi85.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi86(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 86. * (np.pi/180)
	return phi - 86. * (np.pi/180)
isphi86.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi87(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 87. * (np.pi/180)
	return phi - 87. * (np.pi/180)
isphi87.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi88(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 88. * (np.pi/180)
	return phi - 88. * (np.pi/180)
isphi88.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi89(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 89. * (np.pi/180)
	return phi - 89. * (np.pi/180)
isphi89.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi90(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 90. * (np.pi/180)
	return phi - 90. * (np.pi/180)
isphi90.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi91(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 91. * (np.pi/180)
	return phi - 91. * (np.pi/180)
isphi91.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi92(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 92. * (np.pi/180)
	return phi - 92. * (np.pi/180)
isphi92.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi93(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 93. * (np.pi/180)
	return phi - 93. * (np.pi/180)
isphi93.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi94(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 94. * (np.pi/180)
	return phi - 94. * (np.pi/180)
isphi94.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi95(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 95. * (np.pi/180)
	return phi - 95. * (np.pi/180)
isphi95.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi96(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 96. * (np.pi/180)
	return phi - 96. * (np.pi/180)
isphi96.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi97(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 97. * (np.pi/180)
	return phi - 97. * (np.pi/180)
isphi97.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi98(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 98. * (np.pi/180)
	return phi - 98. * (np.pi/180)
isphi98.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi99(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 99. * (np.pi/180)
	return phi - 99. * (np.pi/180)
isphi99.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi100(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 100. * (np.pi/180)
	return phi - 100. * (np.pi/180)
isphi100.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi101(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 101. * (np.pi/180)
	return phi - 101. * (np.pi/180)
isphi101.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi102(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 102. * (np.pi/180)
	return phi - 102. * (np.pi/180)
isphi102.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi103(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 103. * (np.pi/180)
	return phi - 103. * (np.pi/180)
isphi103.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi104(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 104. * (np.pi/180)
	return phi - 104. * (np.pi/180)
isphi104.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi105(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 105. * (np.pi/180)
	return phi - 105. * (np.pi/180)
isphi105.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi106(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 106. * (np.pi/180)
	return phi - 106. * (np.pi/180)
isphi106.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi107(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 107. * (np.pi/180)
	return phi - 107. * (np.pi/180)
isphi107.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi108(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 108. * (np.pi/180)
	return phi - 108. * (np.pi/180)
isphi108.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi109(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 109. * (np.pi/180)
	return phi - 109. * (np.pi/180)
isphi109.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi110(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 110. * (np.pi/180)
	return phi - 110. * (np.pi/180)
isphi110.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi111(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 111. * (np.pi/180)
	return phi - 111. * (np.pi/180)
isphi111.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi112(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 112. * (np.pi/180)
	return phi - 112. * (np.pi/180)
isphi112.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi113(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 113. * (np.pi/180)
	return phi - 113. * (np.pi/180)
isphi113.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi114(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 114. * (np.pi/180)
	return phi - 114. * (np.pi/180)
isphi114.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi115(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 115. * (np.pi/180)
	return phi - 115. * (np.pi/180)
isphi115.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi116(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 116. * (np.pi/180)
	return phi - 116. * (np.pi/180)
isphi116.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi117(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 117. * (np.pi/180)
	return phi - 117. * (np.pi/180)
isphi117.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi118(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 118. * (np.pi/180)
	return phi - 118. * (np.pi/180)
isphi118.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi119(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 119. * (np.pi/180)
	return phi - 119. * (np.pi/180)
isphi119.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi120(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 120. * (np.pi/180)
	return phi - 120. * (np.pi/180)
isphi120.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi121(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 121. * (np.pi/180)
	return phi - 121. * (np.pi/180)
isphi121.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi122(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 122. * (np.pi/180)
	return phi - 122. * (np.pi/180)
isphi122.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi123(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 123. * (np.pi/180)
	return phi - 123. * (np.pi/180)
isphi123.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi124(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 124. * (np.pi/180)
	return phi - 124. * (np.pi/180)
isphi124.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi125(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 125. * (np.pi/180)
	return phi - 125. * (np.pi/180)
isphi125.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi126(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 126. * (np.pi/180)
	return phi - 126. * (np.pi/180)
isphi126.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi127(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 127. * (np.pi/180)
	return phi - 127. * (np.pi/180)
isphi127.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi128(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 128. * (np.pi/180)
	return phi - 128. * (np.pi/180)
isphi128.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi129(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 129. * (np.pi/180)
	return phi - 129. * (np.pi/180)
isphi129.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi130(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 130. * (np.pi/180)
	return phi - 130. * (np.pi/180)
isphi130.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi131(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 131. * (np.pi/180)
	return phi - 131. * (np.pi/180)
isphi131.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi132(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 132. * (np.pi/180)
	return phi - 132. * (np.pi/180)
isphi132.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi133(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 133. * (np.pi/180)
	return phi - 133. * (np.pi/180)
isphi133.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi134(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 134. * (np.pi/180)
	return phi - 134. * (np.pi/180)
isphi134.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi135(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 135. * (np.pi/180)
	return phi - 135. * (np.pi/180)
isphi135.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi136(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 136. * (np.pi/180)
	return phi - 136. * (np.pi/180)
isphi136.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi137(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 137. * (np.pi/180)
	return phi - 137. * (np.pi/180)
isphi137.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi138(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 138. * (np.pi/180)
	return phi - 138. * (np.pi/180)
isphi138.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi139(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 139. * (np.pi/180)
	return phi - 139. * (np.pi/180)
isphi139.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi140(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 140. * (np.pi/180)
	return phi - 140. * (np.pi/180)
isphi140.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi141(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 141. * (np.pi/180)
	return phi - 141. * (np.pi/180)
isphi141.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi142(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 142. * (np.pi/180)
	return phi - 142. * (np.pi/180)
isphi142.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi143(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 143. * (np.pi/180)
	return phi - 143. * (np.pi/180)
isphi143.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi144(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 144. * (np.pi/180)
	return phi - 144. * (np.pi/180)
isphi144.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi145(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 145. * (np.pi/180)
	return phi - 145. * (np.pi/180)
isphi145.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi146(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 146. * (np.pi/180)
	return phi - 146. * (np.pi/180)
isphi146.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi147(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 147. * (np.pi/180)
	return phi - 147. * (np.pi/180)
isphi147.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi148(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 148. * (np.pi/180)
	return phi - 148. * (np.pi/180)
isphi148.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi149(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 149. * (np.pi/180)
	return phi - 149. * (np.pi/180)
isphi149.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi150(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 150. * (np.pi/180)
	return phi - 150. * (np.pi/180)
isphi150.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi151(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 151. * (np.pi/180)
	return phi - 151. * (np.pi/180)
isphi151.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi152(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 152. * (np.pi/180)
	return phi - 152. * (np.pi/180)
isphi152.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi153(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 153. * (np.pi/180)
	return phi - 153. * (np.pi/180)
isphi153.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi154(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 154. * (np.pi/180)
	return phi - 154. * (np.pi/180)
isphi154.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi155(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 155. * (np.pi/180)
	return phi - 155. * (np.pi/180)
isphi155.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi156(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 156. * (np.pi/180)
	return phi - 156. * (np.pi/180)
isphi156.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi157(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 157. * (np.pi/180)
	return phi - 157. * (np.pi/180)
isphi157.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi158(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 158. * (np.pi/180)
	return phi - 158. * (np.pi/180)
isphi158.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi159(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 159. * (np.pi/180)
	return phi - 159. * (np.pi/180)
isphi159.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi160(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 160. * (np.pi/180)
	return phi - 160. * (np.pi/180)
isphi160.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi161(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 161. * (np.pi/180)
	return phi - 161. * (np.pi/180)
isphi161.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi162(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 162. * (np.pi/180)
	return phi - 162. * (np.pi/180)
isphi162.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi163(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 163. * (np.pi/180)
	return phi - 163. * (np.pi/180)
isphi163.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi164(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 164. * (np.pi/180)
	return phi - 164. * (np.pi/180)
isphi164.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi165(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 165. * (np.pi/180)
	return phi - 165. * (np.pi/180)
isphi165.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi166(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 166. * (np.pi/180)
	return phi - 166. * (np.pi/180)
isphi166.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi167(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 167. * (np.pi/180)
	return phi - 167. * (np.pi/180)
isphi167.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi168(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 168. * (np.pi/180)
	return phi - 168. * (np.pi/180)
isphi168.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi169(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 169. * (np.pi/180)
	return phi - 169. * (np.pi/180)
isphi169.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi170(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 170. * (np.pi/180)
	return phi - 170. * (np.pi/180)
isphi170.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi171(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 171. * (np.pi/180)
	return phi - 171. * (np.pi/180)
isphi171.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi172(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 172. * (np.pi/180)
	return phi - 172. * (np.pi/180)
isphi172.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi173(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 173. * (np.pi/180)
	return phi - 173. * (np.pi/180)
isphi173.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi174(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 174. * (np.pi/180)
	return phi - 174. * (np.pi/180)
isphi174.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi175(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 175. * (np.pi/180)
	return phi - 175. * (np.pi/180)
isphi175.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi176(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 176. * (np.pi/180)
	return phi - 176. * (np.pi/180)
isphi176.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi177(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 177. * (np.pi/180)
	return phi - 177. * (np.pi/180)
isphi177.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi178(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 178. * (np.pi/180)
	return phi - 178. * (np.pi/180)
isphi178.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi179(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 179. * (np.pi/180)
	return phi - 179. * (np.pi/180)
isphi179.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi180(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 180. * (np.pi/180)
	return phi - 180. * (np.pi/180)
isphi180.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi181(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 181. * (np.pi/180)
	return phi - 181. * (np.pi/180)
isphi181.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi182(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 182. * (np.pi/180)
	return phi - 182. * (np.pi/180)
isphi182.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi183(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 183. * (np.pi/180)
	return phi - 183. * (np.pi/180)
isphi183.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi184(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 184. * (np.pi/180)
	return phi - 184. * (np.pi/180)
isphi184.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi185(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 185. * (np.pi/180)
	return phi - 185. * (np.pi/180)
isphi185.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi186(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 186. * (np.pi/180)
	return phi - 186. * (np.pi/180)
isphi186.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi187(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 187. * (np.pi/180)
	return phi - 187. * (np.pi/180)
isphi187.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi188(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 188. * (np.pi/180)
	return phi - 188. * (np.pi/180)
isphi188.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi189(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 189. * (np.pi/180)
	return phi - 189. * (np.pi/180)
isphi189.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi190(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 190. * (np.pi/180)
	return phi - 190. * (np.pi/180)
isphi190.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi191(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 191. * (np.pi/180)
	return phi - 191. * (np.pi/180)
isphi191.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi192(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 192. * (np.pi/180)
	return phi - 192. * (np.pi/180)
isphi192.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi193(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 193. * (np.pi/180)
	return phi - 193. * (np.pi/180)
isphi193.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi194(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 194. * (np.pi/180)
	return phi - 194. * (np.pi/180)
isphi194.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi195(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 195. * (np.pi/180)
	return phi - 195. * (np.pi/180)
isphi195.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi196(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 196. * (np.pi/180)
	return phi - 196. * (np.pi/180)
isphi196.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi197(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 197. * (np.pi/180)
	return phi - 197. * (np.pi/180)
isphi197.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi198(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 198. * (np.pi/180)
	return phi - 198. * (np.pi/180)
isphi198.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi199(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 199. * (np.pi/180)
	return phi - 199. * (np.pi/180)
isphi199.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi200(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 200. * (np.pi/180)
	return phi - 200. * (np.pi/180)
isphi200.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi201(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 201. * (np.pi/180)
	return phi - 201. * (np.pi/180)
isphi201.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi202(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 202. * (np.pi/180)
	return phi - 202. * (np.pi/180)
isphi202.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi203(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 203. * (np.pi/180)
	return phi - 203. * (np.pi/180)
isphi203.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi204(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 204. * (np.pi/180)
	return phi - 204. * (np.pi/180)
isphi204.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi205(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 205. * (np.pi/180)
	return phi - 205. * (np.pi/180)
isphi205.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi206(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 206. * (np.pi/180)
	return phi - 206. * (np.pi/180)
isphi206.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi207(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 207. * (np.pi/180)
	return phi - 207. * (np.pi/180)
isphi207.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi208(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 208. * (np.pi/180)
	return phi - 208. * (np.pi/180)
isphi208.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi209(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 209. * (np.pi/180)
	return phi - 209. * (np.pi/180)
isphi209.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi210(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 210. * (np.pi/180)
	return phi - 210. * (np.pi/180)
isphi210.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi211(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 211. * (np.pi/180)
	return phi - 211. * (np.pi/180)
isphi211.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi212(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 212. * (np.pi/180)
	return phi - 212. * (np.pi/180)
isphi212.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi213(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 213. * (np.pi/180)
	return phi - 213. * (np.pi/180)
isphi213.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi214(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 214. * (np.pi/180)
	return phi - 214. * (np.pi/180)
isphi214.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi215(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 215. * (np.pi/180)
	return phi - 215. * (np.pi/180)
isphi215.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi216(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 216. * (np.pi/180)
	return phi - 216. * (np.pi/180)
isphi216.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi217(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 217. * (np.pi/180)
	return phi - 217. * (np.pi/180)
isphi217.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi218(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 218. * (np.pi/180)
	return phi - 218. * (np.pi/180)
isphi218.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi219(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 219. * (np.pi/180)
	return phi - 219. * (np.pi/180)
isphi219.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi220(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 220. * (np.pi/180)
	return phi - 220. * (np.pi/180)
isphi220.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi221(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 221. * (np.pi/180)
	return phi - 221. * (np.pi/180)
isphi221.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi222(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 222. * (np.pi/180)
	return phi - 222. * (np.pi/180)
isphi222.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi223(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 223. * (np.pi/180)
	return phi - 223. * (np.pi/180)
isphi223.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi224(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 224. * (np.pi/180)
	return phi - 224. * (np.pi/180)
isphi224.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi225(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 225. * (np.pi/180)
	return phi - 225. * (np.pi/180)
isphi225.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi226(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 226. * (np.pi/180)
	return phi - 226. * (np.pi/180)
isphi226.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi227(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 227. * (np.pi/180)
	return phi - 227. * (np.pi/180)
isphi227.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi228(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 228. * (np.pi/180)
	return phi - 228. * (np.pi/180)
isphi228.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi229(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 229. * (np.pi/180)
	return phi - 229. * (np.pi/180)
isphi229.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi230(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 230. * (np.pi/180)
	return phi - 230. * (np.pi/180)
isphi230.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi231(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 231. * (np.pi/180)
	return phi - 231. * (np.pi/180)
isphi231.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi232(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 232. * (np.pi/180)
	return phi - 232. * (np.pi/180)
isphi232.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi233(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 233. * (np.pi/180)
	return phi - 233. * (np.pi/180)
isphi233.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi234(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 234. * (np.pi/180)
	return phi - 234. * (np.pi/180)
isphi234.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi235(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 235. * (np.pi/180)
	return phi - 235. * (np.pi/180)
isphi235.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi236(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 236. * (np.pi/180)
	return phi - 236. * (np.pi/180)
isphi236.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi237(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 237. * (np.pi/180)
	return phi - 237. * (np.pi/180)
isphi237.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi238(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 238. * (np.pi/180)
	return phi - 238. * (np.pi/180)
isphi238.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi239(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 239. * (np.pi/180)
	return phi - 239. * (np.pi/180)
isphi239.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi240(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 240. * (np.pi/180)
	return phi - 240. * (np.pi/180)
isphi240.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi241(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 241. * (np.pi/180)
	return phi - 241. * (np.pi/180)
isphi241.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi242(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 242. * (np.pi/180)
	return phi - 242. * (np.pi/180)
isphi242.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi243(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 243. * (np.pi/180)
	return phi - 243. * (np.pi/180)
isphi243.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi244(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 244. * (np.pi/180)
	return phi - 244. * (np.pi/180)
isphi244.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi245(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 245. * (np.pi/180)
	return phi - 245. * (np.pi/180)
isphi245.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi246(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 246. * (np.pi/180)
	return phi - 246. * (np.pi/180)
isphi246.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi247(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 247. * (np.pi/180)
	return phi - 247. * (np.pi/180)
isphi247.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi248(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 248. * (np.pi/180)
	return phi - 248. * (np.pi/180)
isphi248.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi249(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 249. * (np.pi/180)
	return phi - 249. * (np.pi/180)
isphi249.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi250(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 250. * (np.pi/180)
	return phi - 250. * (np.pi/180)
isphi250.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi251(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 251. * (np.pi/180)
	return phi - 251. * (np.pi/180)
isphi251.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi252(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 252. * (np.pi/180)
	return phi - 252. * (np.pi/180)
isphi252.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi253(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 253. * (np.pi/180)
	return phi - 253. * (np.pi/180)
isphi253.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi254(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 254. * (np.pi/180)
	return phi - 254. * (np.pi/180)
isphi254.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi255(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 255. * (np.pi/180)
	return phi - 255. * (np.pi/180)
isphi255.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi256(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 256. * (np.pi/180)
	return phi - 256. * (np.pi/180)
isphi256.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi257(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 257. * (np.pi/180)
	return phi - 257. * (np.pi/180)
isphi257.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi258(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 258. * (np.pi/180)
	return phi - 258. * (np.pi/180)
isphi258.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi259(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 259. * (np.pi/180)
	return phi - 259. * (np.pi/180)
isphi259.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi260(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 260. * (np.pi/180)
	return phi - 260. * (np.pi/180)
isphi260.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi261(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 261. * (np.pi/180)
	return phi - 261. * (np.pi/180)
isphi261.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi262(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 262. * (np.pi/180)
	return phi - 262. * (np.pi/180)
isphi262.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi263(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 263. * (np.pi/180)
	return phi - 263. * (np.pi/180)
isphi263.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi264(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 264. * (np.pi/180)
	return phi - 264. * (np.pi/180)
isphi264.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi265(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 265. * (np.pi/180)
	return phi - 265. * (np.pi/180)
isphi265.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi266(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 266. * (np.pi/180)
	return phi - 266. * (np.pi/180)
isphi266.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi267(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 267. * (np.pi/180)
	return phi - 267. * (np.pi/180)
isphi267.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi268(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 268. * (np.pi/180)
	return phi - 268. * (np.pi/180)
isphi268.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi269(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 269. * (np.pi/180)
	return phi - 269. * (np.pi/180)
isphi269.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi270(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 270. * (np.pi/180)
	return phi - 270. * (np.pi/180)
isphi270.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi271(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 271. * (np.pi/180)
	return phi - 271. * (np.pi/180)
isphi271.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi272(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 272. * (np.pi/180)
	return phi - 272. * (np.pi/180)
isphi272.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi273(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 273. * (np.pi/180)
	return phi - 273. * (np.pi/180)
isphi273.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi274(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 274. * (np.pi/180)
	return phi - 274. * (np.pi/180)
isphi274.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi275(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 275. * (np.pi/180)
	return phi - 275. * (np.pi/180)
isphi275.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi276(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 276. * (np.pi/180)
	return phi - 276. * (np.pi/180)
isphi276.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi277(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 277. * (np.pi/180)
	return phi - 277. * (np.pi/180)
isphi277.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi278(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 278. * (np.pi/180)
	return phi - 278. * (np.pi/180)
isphi278.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi279(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 279. * (np.pi/180)
	return phi - 279. * (np.pi/180)
isphi279.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi280(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 280. * (np.pi/180)
	return phi - 280. * (np.pi/180)
isphi280.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi281(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 281. * (np.pi/180)
	return phi - 281. * (np.pi/180)
isphi281.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi282(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 282. * (np.pi/180)
	return phi - 282. * (np.pi/180)
isphi282.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi283(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 283. * (np.pi/180)
	return phi - 283. * (np.pi/180)
isphi283.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi284(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 284. * (np.pi/180)
	return phi - 284. * (np.pi/180)
isphi284.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi285(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 285. * (np.pi/180)
	return phi - 285. * (np.pi/180)
isphi285.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi286(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 286. * (np.pi/180)
	return phi - 286. * (np.pi/180)
isphi286.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi287(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 287. * (np.pi/180)
	return phi - 287. * (np.pi/180)
isphi287.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi288(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 288. * (np.pi/180)
	return phi - 288. * (np.pi/180)
isphi288.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi289(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 289. * (np.pi/180)
	return phi - 289. * (np.pi/180)
isphi289.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi290(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 290. * (np.pi/180)
	return phi - 290. * (np.pi/180)
isphi290.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi291(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 291. * (np.pi/180)
	return phi - 291. * (np.pi/180)
isphi291.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi292(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 292. * (np.pi/180)
	return phi - 292. * (np.pi/180)
isphi292.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi293(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 293. * (np.pi/180)
	return phi - 293. * (np.pi/180)
isphi293.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi294(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 294. * (np.pi/180)
	return phi - 294. * (np.pi/180)
isphi294.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi295(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 295. * (np.pi/180)
	return phi - 295. * (np.pi/180)
isphi295.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi296(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 296. * (np.pi/180)
	return phi - 296. * (np.pi/180)
isphi296.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi297(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 297. * (np.pi/180)
	return phi - 297. * (np.pi/180)
isphi297.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi298(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 298. * (np.pi/180)
	return phi - 298. * (np.pi/180)
isphi298.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi299(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 299. * (np.pi/180)
	return phi - 299. * (np.pi/180)
isphi299.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi300(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 300. * (np.pi/180)
	return phi - 300. * (np.pi/180)
isphi300.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi301(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 301. * (np.pi/180)
	return phi - 301. * (np.pi/180)
isphi301.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi302(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 302. * (np.pi/180)
	return phi - 302. * (np.pi/180)
isphi302.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi303(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 303. * (np.pi/180)
	return phi - 303. * (np.pi/180)
isphi303.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi304(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 304. * (np.pi/180)
	return phi - 304. * (np.pi/180)
isphi304.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi305(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 305. * (np.pi/180)
	return phi - 305. * (np.pi/180)
isphi305.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi306(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 306. * (np.pi/180)
	return phi - 306. * (np.pi/180)
isphi306.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi307(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 307. * (np.pi/180)
	return phi - 307. * (np.pi/180)
isphi307.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi308(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 308. * (np.pi/180)
	return phi - 308. * (np.pi/180)
isphi308.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi309(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 309. * (np.pi/180)
	return phi - 309. * (np.pi/180)
isphi309.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi310(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 310. * (np.pi/180)
	return phi - 310. * (np.pi/180)
isphi310.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi311(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 311. * (np.pi/180)
	return phi - 311. * (np.pi/180)
isphi311.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi312(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 312. * (np.pi/180)
	return phi - 312. * (np.pi/180)
isphi312.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi313(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 313. * (np.pi/180)
	return phi - 313. * (np.pi/180)
isphi313.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi314(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 314. * (np.pi/180)
	return phi - 314. * (np.pi/180)
isphi314.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi315(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 315. * (np.pi/180)
	return phi - 315. * (np.pi/180)
isphi315.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi316(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 316. * (np.pi/180)
	return phi - 316. * (np.pi/180)
isphi316.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi317(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 317. * (np.pi/180)
	return phi - 317. * (np.pi/180)
isphi317.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi318(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 318. * (np.pi/180)
	return phi - 318. * (np.pi/180)
isphi318.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi319(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 319. * (np.pi/180)
	return phi - 319. * (np.pi/180)
isphi319.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi320(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 320. * (np.pi/180)
	return phi - 320. * (np.pi/180)
isphi320.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi321(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 321. * (np.pi/180)
	return phi - 321. * (np.pi/180)
isphi321.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi322(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 322. * (np.pi/180)
	return phi - 322. * (np.pi/180)
isphi322.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi323(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 323. * (np.pi/180)
	return phi - 323. * (np.pi/180)
isphi323.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi324(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 324. * (np.pi/180)
	return phi - 324. * (np.pi/180)
isphi324.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi325(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 325. * (np.pi/180)
	return phi - 325. * (np.pi/180)
isphi325.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi326(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 326. * (np.pi/180)
	return phi - 326. * (np.pi/180)
isphi326.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi327(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 327. * (np.pi/180)
	return phi - 327. * (np.pi/180)
isphi327.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi328(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 328. * (np.pi/180)
	return phi - 328. * (np.pi/180)
isphi328.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi329(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 329. * (np.pi/180)
	return phi - 329. * (np.pi/180)
isphi329.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi330(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 330. * (np.pi/180)
	return phi - 330. * (np.pi/180)
isphi330.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi331(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 331. * (np.pi/180)
	return phi - 331. * (np.pi/180)
isphi331.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi332(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 332. * (np.pi/180)
	return phi - 332. * (np.pi/180)
isphi332.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi333(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 333. * (np.pi/180)
	return phi - 333. * (np.pi/180)
isphi333.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi334(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 334. * (np.pi/180)
	return phi - 334. * (np.pi/180)
isphi334.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi335(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 335. * (np.pi/180)
	return phi - 335. * (np.pi/180)
isphi335.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi336(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 336. * (np.pi/180)
	return phi - 336. * (np.pi/180)
isphi336.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi337(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 337. * (np.pi/180)
	return phi - 337. * (np.pi/180)
isphi337.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi338(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 338. * (np.pi/180)
	return phi - 338. * (np.pi/180)
isphi338.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi339(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 339. * (np.pi/180)
	return phi - 339. * (np.pi/180)
isphi339.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi340(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 340. * (np.pi/180)
	return phi - 340. * (np.pi/180)
isphi340.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi341(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 341. * (np.pi/180)
	return phi - 341. * (np.pi/180)
isphi341.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi342(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 342. * (np.pi/180)
	return phi - 342. * (np.pi/180)
isphi342.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi343(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 343. * (np.pi/180)
	return phi - 343. * (np.pi/180)
isphi343.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi344(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 344. * (np.pi/180)
	return phi - 344. * (np.pi/180)
isphi344.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi345(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 345. * (np.pi/180)
	return phi - 345. * (np.pi/180)
isphi345.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi346(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 346. * (np.pi/180)
	return phi - 346. * (np.pi/180)
isphi346.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi347(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 347. * (np.pi/180)
	return phi - 347. * (np.pi/180)
isphi347.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi348(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 348. * (np.pi/180)
	return phi - 348. * (np.pi/180)
isphi348.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi349(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 349. * (np.pi/180)
	return phi - 349. * (np.pi/180)
isphi349.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi350(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 350. * (np.pi/180)
	return phi - 350. * (np.pi/180)
isphi350.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi351(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 351. * (np.pi/180)
	return phi - 351. * (np.pi/180)
isphi351.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi352(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 352. * (np.pi/180)
	return phi - 352. * (np.pi/180)
isphi352.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi353(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 353. * (np.pi/180)
	return phi - 353. * (np.pi/180)
isphi353.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi354(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 354. * (np.pi/180)
	return phi - 354. * (np.pi/180)
isphi354.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi355(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 355. * (np.pi/180)
	return phi - 355. * (np.pi/180)
isphi355.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi356(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 356. * (np.pi/180)
	return phi - 356. * (np.pi/180)
isphi356.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi357(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 357. * (np.pi/180)
	return phi - 357. * (np.pi/180)
isphi357.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi358(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 358. * (np.pi/180)
	return phi - 358. * (np.pi/180)
isphi358.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi359(t, p_XYZ, Mesh):
	#p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	phi = np.where(phi<0, phi+2*np.pi, phi)
	#return p_RTP[2] - 359. * (np.pi/180)
	return phi - 359. * (np.pi/180)
isphi359.direction = 1.0

#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi360(t, p_XYZ, Mesh):
	return p_XYZ[1]
isphi360.direction = -1.0

def eventsAndRange(): 
    poincare_events = [ phi_events.inVV, 
                        phi_events.isphi1, 
                        phi_events.isphi2, 
                        phi_events.isphi3, 
                        phi_events.isphi4, 
                        phi_events.isphi5, 
                        phi_events.isphi6, 
                        phi_events.isphi7, 
                        phi_events.isphi8, 
                        phi_events.isphi9, 
                        phi_events.isphi10, 
                        phi_events.isphi11, 
                        phi_events.isphi12, 
                        phi_events.isphi13, 
                        phi_events.isphi14, 
                        phi_events.isphi15, 
                        phi_events.isphi16, 
                        phi_events.isphi17, 
                        phi_events.isphi18, 
                        phi_events.isphi19, 
                        phi_events.isphi20, 
                        phi_events.isphi21, 
                        phi_events.isphi22, 
                        phi_events.isphi23, 
                        phi_events.isphi24, 
                        phi_events.isphi25, 
                        phi_events.isphi26, 
                        phi_events.isphi27, 
                        phi_events.isphi28, 
                        phi_events.isphi29, 
                        phi_events.isphi30, 
                        phi_events.isphi31, 
                        phi_events.isphi32, 
                        phi_events.isphi33, 
                        phi_events.isphi34, 
                        phi_events.isphi35, 
                        phi_events.isphi36, 
                        phi_events.isphi37, 
                        phi_events.isphi38, 
                        phi_events.isphi39, 
                        phi_events.isphi40, 
                        phi_events.isphi41, 
                        phi_events.isphi42, 
                        phi_events.isphi43, 
                        phi_events.isphi44, 
                        phi_events.isphi45, 
                        phi_events.isphi46, 
                        phi_events.isphi47, 
                        phi_events.isphi48, 
                        phi_events.isphi49, 
                        phi_events.isphi50, 
                        phi_events.isphi51, 
                        phi_events.isphi52, 
                        phi_events.isphi53, 
                        phi_events.isphi54, 
                        phi_events.isphi55, 
                        phi_events.isphi56, 
                        phi_events.isphi57, 
                        phi_events.isphi58, 
                        phi_events.isphi59, 
                        phi_events.isphi60, 
                        phi_events.isphi61, 
                        phi_events.isphi62, 
                        phi_events.isphi63, 
                        phi_events.isphi64, 
                        phi_events.isphi65, 
                        phi_events.isphi66, 
                        phi_events.isphi67, 
                        phi_events.isphi68, 
                        phi_events.isphi69, 
                        phi_events.isphi70, 
                        phi_events.isphi71, 
                        phi_events.isphi72, 
                        phi_events.isphi73, 
                        phi_events.isphi74, 
                        phi_events.isphi75, 
                        phi_events.isphi76, 
                        phi_events.isphi77, 
                        phi_events.isphi78, 
                        phi_events.isphi79, 
                        phi_events.isphi80, 
                        phi_events.isphi81, 
                        phi_events.isphi82, 
                        phi_events.isphi83, 
                        phi_events.isphi84, 
                        phi_events.isphi85, 
                        phi_events.isphi86, 
                        phi_events.isphi87, 
                        phi_events.isphi88, 
                        phi_events.isphi89, 
                        phi_events.isphi90, 
                        phi_events.isphi91, 
                        phi_events.isphi92, 
                        phi_events.isphi93, 
                        phi_events.isphi94, 
                        phi_events.isphi95, 
                        phi_events.isphi96, 
                        phi_events.isphi97, 
                        phi_events.isphi98, 
                        phi_events.isphi99, 
                        phi_events.isphi100, 
                        phi_events.isphi101, 
                        phi_events.isphi102, 
                        phi_events.isphi103, 
                        phi_events.isphi104, 
                        phi_events.isphi105, 
                        phi_events.isphi106, 
                        phi_events.isphi107, 
                        phi_events.isphi108, 
                        phi_events.isphi109, 
                        phi_events.isphi110, 
                        phi_events.isphi111, 
                        phi_events.isphi112, 
                        phi_events.isphi113, 
                        phi_events.isphi114, 
                        phi_events.isphi115, 
                        phi_events.isphi116, 
                        phi_events.isphi117, 
                        phi_events.isphi118, 
                        phi_events.isphi119, 
                        phi_events.isphi120, 
                        phi_events.isphi121, 
                        phi_events.isphi122, 
                        phi_events.isphi123, 
                        phi_events.isphi124, 
                        phi_events.isphi125, 
                        phi_events.isphi126, 
                        phi_events.isphi127, 
                        phi_events.isphi128, 
                        phi_events.isphi129, 
                        phi_events.isphi130, 
                        phi_events.isphi131, 
                        phi_events.isphi132, 
                        phi_events.isphi133, 
                        phi_events.isphi134, 
                        phi_events.isphi135, 
                        phi_events.isphi136, 
                        phi_events.isphi137, 
                        phi_events.isphi138, 
                        phi_events.isphi139, 
                        phi_events.isphi140, 
                        phi_events.isphi141, 
                        phi_events.isphi142, 
                        phi_events.isphi143, 
                        phi_events.isphi144, 
                        phi_events.isphi145, 
                        phi_events.isphi146, 
                        phi_events.isphi147, 
                        phi_events.isphi148, 
                        phi_events.isphi149, 
                        phi_events.isphi150, 
                        phi_events.isphi151, 
                        phi_events.isphi152, 
                        phi_events.isphi153, 
                        phi_events.isphi154, 
                        phi_events.isphi155, 
                        phi_events.isphi156, 
                        phi_events.isphi157, 
                        phi_events.isphi158, 
                        phi_events.isphi159, 
                        phi_events.isphi160, 
                        phi_events.isphi161, 
                        phi_events.isphi162, 
                        phi_events.isphi163, 
                        phi_events.isphi164, 
                        phi_events.isphi165, 
                        phi_events.isphi166, 
                        phi_events.isphi167, 
                        phi_events.isphi168, 
                        phi_events.isphi169, 
                        phi_events.isphi170, 
                        phi_events.isphi171, 
                        phi_events.isphi172, 
                        phi_events.isphi173, 
                        phi_events.isphi174, 
                        phi_events.isphi175, 
                        phi_events.isphi176, 
                        phi_events.isphi177, 
                        phi_events.isphi178, 
                        phi_events.isphi179, 
                        phi_events.isphi180, 
                        phi_events.isphi181, 
                        phi_events.isphi182, 
                        phi_events.isphi183, 
                        phi_events.isphi184, 
                        phi_events.isphi185, 
                        phi_events.isphi186, 
                        phi_events.isphi187, 
                        phi_events.isphi188, 
                        phi_events.isphi189, 
                        phi_events.isphi190, 
                        phi_events.isphi191, 
                        phi_events.isphi192, 
                        phi_events.isphi193, 
                        phi_events.isphi194, 
                        phi_events.isphi195, 
                        phi_events.isphi196, 
                        phi_events.isphi197, 
                        phi_events.isphi198, 
                        phi_events.isphi199, 
                        phi_events.isphi200, 
                        phi_events.isphi201, 
                        phi_events.isphi202, 
                        phi_events.isphi203, 
                        phi_events.isphi204, 
                        phi_events.isphi205, 
                        phi_events.isphi206, 
                        phi_events.isphi207, 
                        phi_events.isphi208, 
                        phi_events.isphi209, 
                        phi_events.isphi210, 
                        phi_events.isphi211, 
                        phi_events.isphi212, 
                        phi_events.isphi213, 
                        phi_events.isphi214, 
                        phi_events.isphi215, 
                        phi_events.isphi216, 
                        phi_events.isphi217, 
                        phi_events.isphi218, 
                        phi_events.isphi219, 
                        phi_events.isphi220, 
                        phi_events.isphi221, 
                        phi_events.isphi222, 
                        phi_events.isphi223, 
                        phi_events.isphi224, 
                        phi_events.isphi225, 
                        phi_events.isphi226, 
                        phi_events.isphi227, 
                        phi_events.isphi228, 
                        phi_events.isphi229, 
                        phi_events.isphi230, 
                        phi_events.isphi231, 
                        phi_events.isphi232, 
                        phi_events.isphi233, 
                        phi_events.isphi234, 
                        phi_events.isphi235, 
                        phi_events.isphi236, 
                        phi_events.isphi237, 
                        phi_events.isphi238, 
                        phi_events.isphi239, 
                        phi_events.isphi240, 
                        phi_events.isphi241, 
                        phi_events.isphi242, 
                        phi_events.isphi243, 
                        phi_events.isphi244, 
                        phi_events.isphi245, 
                        phi_events.isphi246, 
                        phi_events.isphi247, 
                        phi_events.isphi248, 
                        phi_events.isphi249, 
                        phi_events.isphi250, 
                        phi_events.isphi251, 
                        phi_events.isphi252, 
                        phi_events.isphi253, 
                        phi_events.isphi254, 
                        phi_events.isphi255, 
                        phi_events.isphi256, 
                        phi_events.isphi257, 
                        phi_events.isphi258, 
                        phi_events.isphi259, 
                        phi_events.isphi260, 
                        phi_events.isphi261, 
                        phi_events.isphi262, 
                        phi_events.isphi263, 
                        phi_events.isphi264, 
                        phi_events.isphi265, 
                        phi_events.isphi266, 
                        phi_events.isphi267, 
                        phi_events.isphi268, 
                        phi_events.isphi269, 
                        phi_events.isphi270, 
                        phi_events.isphi271, 
                        phi_events.isphi272, 
                        phi_events.isphi273, 
                        phi_events.isphi274, 
                        phi_events.isphi275, 
                        phi_events.isphi276, 
                        phi_events.isphi277, 
                        phi_events.isphi278, 
                        phi_events.isphi279, 
                        phi_events.isphi280, 
                        phi_events.isphi281, 
                        phi_events.isphi282, 
                        phi_events.isphi283, 
                        phi_events.isphi284, 
                        phi_events.isphi285, 
                        phi_events.isphi286, 
                        phi_events.isphi287, 
                        phi_events.isphi288, 
                        phi_events.isphi289, 
                        phi_events.isphi290, 
                        phi_events.isphi291, 
                        phi_events.isphi292, 
                        phi_events.isphi293, 
                        phi_events.isphi294, 
                        phi_events.isphi295, 
                        phi_events.isphi296, 
                        phi_events.isphi297, 
                        phi_events.isphi298, 
                        phi_events.isphi299, 
                        phi_events.isphi300, 
                        phi_events.isphi301, 
                        phi_events.isphi302, 
                        phi_events.isphi303, 
                        phi_events.isphi304, 
                        phi_events.isphi305, 
                        phi_events.isphi306, 
                        phi_events.isphi307, 
                        phi_events.isphi308, 
                        phi_events.isphi309, 
                        phi_events.isphi310, 
                        phi_events.isphi311, 
                        phi_events.isphi312, 
                        phi_events.isphi313, 
                        phi_events.isphi314, 
                        phi_events.isphi315, 
                        phi_events.isphi316, 
                        phi_events.isphi317, 
                        phi_events.isphi318, 
                        phi_events.isphi319, 
                        phi_events.isphi320, 
                        phi_events.isphi321, 
                        phi_events.isphi322, 
                        phi_events.isphi323, 
                        phi_events.isphi324, 
                        phi_events.isphi325, 
                        phi_events.isphi326, 
                        phi_events.isphi327, 
                        phi_events.isphi328, 
                        phi_events.isphi329, 
                        phi_events.isphi330, 
                        phi_events.isphi331, 
                        phi_events.isphi332, 
                        phi_events.isphi333, 
                        phi_events.isphi334, 
                        phi_events.isphi335, 
                        phi_events.isphi336, 
                        phi_events.isphi337, 
                        phi_events.isphi338, 
                        phi_events.isphi339, 
                        phi_events.isphi340, 
                        phi_events.isphi341, 
                        phi_events.isphi342, 
                        phi_events.isphi343, 
                        phi_events.isphi344, 
                        phi_events.isphi345, 
                        phi_events.isphi346, 
                        phi_events.isphi347, 
                        phi_events.isphi348, 
                        phi_events.isphi349, 
                        phi_events.isphi350, 
                        phi_events.isphi351, 
                        phi_events.isphi352, 
                        phi_events.isphi353, 
                        phi_events.isphi354, 
                        phi_events.isphi355, 
                        phi_events.isphi356, 
                        phi_events.isphi357, 
                        phi_events.isphi358, 
                        phi_events.isphi359, 
                        phi_events.isphi360] 
    phi_range = np.linspace(np.pi/180., 2*np.pi, 360) 
    return poincare_events,phi_range 
