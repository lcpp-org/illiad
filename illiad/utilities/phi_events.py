import numpy as np
#import numba as nb
from .coordtrans import XYZ_to_RTP
from classes.mesh import *


def inVV(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return Mesh.a - p_RTP[0]
inVV.direction = -1.0
inVV.terminal = True


def isphi1(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 1. * (np.pi/180)

isphi1.direction = 1.0

def isphi2(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 2. * (np.pi/180)

isphi2.direction = 1.0

def isphi3(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 3. * (np.pi/180)
isphi3.direction = 1.0

def isphi4(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 4. * (np.pi/180)
isphi4.direction = 1.0

def isphi5(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 5. * (np.pi/180)
isphi5.direction = 1.0

def isphi6(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 6. * (np.pi/180)
isphi6.direction = 1.0

def isphi7(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 7. * (np.pi/180)
isphi7.direction = 1.0

def isphi8(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 8. * (np.pi/180)
isphi8.direction = 1.0

def isphi9(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 9. * (np.pi/180)
isphi9.direction = 1.0

def isphi10(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 10. * (np.pi/180)
isphi10.direction = 1.0

def isphi11(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 11. * (np.pi/180)
isphi11.direction = 1.0

def isphi12(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 12. * (np.pi/180)
isphi12.direction = 1.0

def isphi13(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 13. * (np.pi/180)
isphi13.direction = 1.0

def isphi14(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 14. * (np.pi/180)
isphi14.direction = 1.0

def isphi15(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi

	return phi - 15. * (np.pi/180)
isphi15.direction = 1.0

def isphi16(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 16. * (np.pi/180)
isphi16.direction = 1.0

def isphi17(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 17. * (np.pi/180)
isphi17.direction = 1.0

def isphi18(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 18. * (np.pi/180)
isphi18.direction = 1.0

def isphi19(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 19. * (np.pi/180)
isphi19.direction = 1.0

def isphi20(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 20. * (np.pi/180)
isphi20.direction = 1.0

def isphi21(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 21. * (np.pi/180)
isphi21.direction = 1.0

def isphi22(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 22. * (np.pi/180)
isphi22.direction = 1.0

def isphi23(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 23. * (np.pi/180)
isphi23.direction = 1.0

def isphi24(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 24. * (np.pi/180)
isphi24.direction = 1.0

def isphi25(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 25. * (np.pi/180)
isphi25.direction = 1.0

def isphi26(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 26. * (np.pi/180)
isphi26.direction = 1.0

def isphi27(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 27. * (np.pi/180)
isphi27.direction = 1.0

def isphi28(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 28. * (np.pi/180)
isphi28.direction = 1.0

def isphi29(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 29. * (np.pi/180)
isphi29.direction = 1.0

def isphi30(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 30. * (np.pi/180)
isphi30.direction = 1.0

def isphi31(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 31. * (np.pi/180)
isphi31.direction = 1.0

def isphi32(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 32. * (np.pi/180)
isphi32.direction = 1.0

def isphi33(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 33. * (np.pi/180)
isphi33.direction = 1.0

def isphi34(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 34. * (np.pi/180)
isphi34.direction = 1.0

def isphi35(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 35. * (np.pi/180)
isphi35.direction = 1.0

def isphi36(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 36. * (np.pi/180)
isphi36.direction = 1.0

def isphi37(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 37. * (np.pi/180)
isphi37.direction = 1.0

def isphi38(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 38. * (np.pi/180)
isphi38.direction = 1.0

def isphi39(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 39. * (np.pi/180)
isphi39.direction = 1.0

def isphi40(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 40. * (np.pi/180)
isphi40.direction = 1.0

def isphi41(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 41. * (np.pi/180)
isphi41.direction = 1.0

def isphi42(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 42. * (np.pi/180)
isphi42.direction = 1.0

def isphi43(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 43. * (np.pi/180)
isphi43.direction = 1.0

def isphi44(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 44. * (np.pi/180)
isphi44.direction = 1.0

def isphi45(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 45. * (np.pi/180)
isphi45.direction = 1.0

def isphi46(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 46. * (np.pi/180)
isphi46.direction = 1.0

def isphi47(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 47. * (np.pi/180)
isphi47.direction = 1.0

def isphi48(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 48. * (np.pi/180)
isphi48.direction = 1.0

def isphi49(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 49. * (np.pi/180)
isphi49.direction = 1.0

def isphi50(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 50. * (np.pi/180)
isphi50.direction = 1.0

def isphi51(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 51. * (np.pi/180)
isphi51.direction = 1.0

def isphi52(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 52. * (np.pi/180)
isphi52.direction = 1.0

def isphi53(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 53. * (np.pi/180)
isphi53.direction = 1.0

def isphi54(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 54. * (np.pi/180)
isphi54.direction = 1.0

def isphi55(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 55. * (np.pi/180)
isphi55.direction = 1.0

def isphi56(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 56. * (np.pi/180)
isphi56.direction = 1.0

def isphi57(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 57. * (np.pi/180)
isphi57.direction = 1.0

def isphi58(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 58. * (np.pi/180)
isphi58.direction = 1.0

def isphi59(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 59. * (np.pi/180)
isphi59.direction = 1.0

def isphi60(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 60. * (np.pi/180)
isphi60.direction = 1.0

def isphi61(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 61. * (np.pi/180)
isphi61.direction = 1.0

def isphi62(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 62. * (np.pi/180)
isphi62.direction = 1.0

def isphi63(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 63. * (np.pi/180)
isphi63.direction = 1.0

def isphi64(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 64. * (np.pi/180)
isphi64.direction = 1.0

def isphi65(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 65. * (np.pi/180)
isphi65.direction = 1.0

def isphi66(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 66. * (np.pi/180)
isphi66.direction = 1.0

def isphi67(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 67. * (np.pi/180)
isphi67.direction = 1.0

def isphi68(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 68. * (np.pi/180)
isphi68.direction = 1.0

def isphi69(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 69. * (np.pi/180)
isphi69.direction = 1.0

def isphi70(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 70. * (np.pi/180)
isphi70.direction = 1.0

def isphi71(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 71. * (np.pi/180)
isphi71.direction = 1.0

def isphi72(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 72. * (np.pi/180)
isphi72.direction = 1.0

def isphi73(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 73. * (np.pi/180)
isphi73.direction = 1.0

def isphi74(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 74. * (np.pi/180)
isphi74.direction = 1.0

def isphi75(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 75. * (np.pi/180)
isphi75.direction = 1.0

def isphi76(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 76. * (np.pi/180)
isphi76.direction = 1.0

def isphi77(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 77. * (np.pi/180)
isphi77.direction = 1.0

def isphi78(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 78. * (np.pi/180)
isphi78.direction = 1.0

def isphi79(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 79. * (np.pi/180)
isphi79.direction = 1.0

def isphi80(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 80. * (np.pi/180)
isphi80.direction = 1.0

def isphi81(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 81. * (np.pi/180)
isphi81.direction = 1.0

def isphi82(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 82. * (np.pi/180)
isphi82.direction = 1.0

def isphi83(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 83. * (np.pi/180)
isphi83.direction = 1.0

def isphi84(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 84. * (np.pi/180)
isphi84.direction = 1.0

def isphi85(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 85. * (np.pi/180)
isphi85.direction = 1.0

def isphi86(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 86. * (np.pi/180)
isphi86.direction = 1.0

def isphi87(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 87. * (np.pi/180)
isphi87.direction = 1.0

def isphi88(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 88. * (np.pi/180)
isphi88.direction = 1.0

def isphi89(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 89. * (np.pi/180)
isphi89.direction = 1.0

def isphi90(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 90. * (np.pi/180)
isphi90.direction = 1.0

def isphi91(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 91. * (np.pi/180)
isphi91.direction = 1.0

def isphi92(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 92. * (np.pi/180)
isphi92.direction = 1.0

def isphi93(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 93. * (np.pi/180)
isphi93.direction = 1.0

def isphi94(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 94. * (np.pi/180)
isphi94.direction = 1.0

def isphi95(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 95. * (np.pi/180)
isphi95.direction = 1.0

def isphi96(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 96. * (np.pi/180)
isphi96.direction = 1.0

def isphi97(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 97. * (np.pi/180)
isphi97.direction = 1.0

def isphi98(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 98. * (np.pi/180)
isphi98.direction = 1.0

def isphi99(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 99. * (np.pi/180)
isphi99.direction = 1.0

def isphi100(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 100. * (np.pi/180)
isphi100.direction = 1.0

def isphi101(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 101. * (np.pi/180)
isphi101.direction = 1.0

def isphi102(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 102. * (np.pi/180)
isphi102.direction = 1.0

def isphi103(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 103. * (np.pi/180)
isphi103.direction = 1.0

def isphi104(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 104. * (np.pi/180)
isphi104.direction = 1.0

def isphi105(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 105. * (np.pi/180)
isphi105.direction = 1.0

def isphi106(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 106. * (np.pi/180)
isphi106.direction = 1.0

def isphi107(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 107. * (np.pi/180)
isphi107.direction = 1.0

def isphi108(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 108. * (np.pi/180)
isphi108.direction = 1.0

def isphi109(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 109. * (np.pi/180)
isphi109.direction = 1.0

def isphi110(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 110. * (np.pi/180)
isphi110.direction = 1.0

def isphi111(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 111. * (np.pi/180)
isphi111.direction = 1.0

def isphi112(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 112. * (np.pi/180)
isphi112.direction = 1.0

def isphi113(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 113. * (np.pi/180)
isphi113.direction = 1.0

def isphi114(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 114. * (np.pi/180)
isphi114.direction = 1.0

def isphi115(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 115. * (np.pi/180)
isphi115.direction = 1.0

def isphi116(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 116. * (np.pi/180)
isphi116.direction = 1.0

def isphi117(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 117. * (np.pi/180)
isphi117.direction = 1.0

def isphi118(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 118. * (np.pi/180)
isphi118.direction = 1.0

def isphi119(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 119. * (np.pi/180)
isphi119.direction = 1.0

def isphi120(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 120. * (np.pi/180)
isphi120.direction = 1.0

def isphi121(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 121. * (np.pi/180)
isphi121.direction = 1.0

def isphi122(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 122. * (np.pi/180)
isphi122.direction = 1.0

def isphi123(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 123. * (np.pi/180)
isphi123.direction = 1.0

def isphi124(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 124. * (np.pi/180)
isphi124.direction = 1.0

def isphi125(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 125. * (np.pi/180)
isphi125.direction = 1.0

def isphi126(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 126. * (np.pi/180)
isphi126.direction = 1.0

def isphi127(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 127. * (np.pi/180)
isphi127.direction = 1.0

def isphi128(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 128. * (np.pi/180)
isphi128.direction = 1.0

def isphi129(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 129. * (np.pi/180)
isphi129.direction = 1.0

def isphi130(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 130. * (np.pi/180)
isphi130.direction = 1.0

def isphi131(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 131. * (np.pi/180)
isphi131.direction = 1.0

def isphi132(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 132. * (np.pi/180)
isphi132.direction = 1.0

def isphi133(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 133. * (np.pi/180)
isphi133.direction = 1.0

def isphi134(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 134. * (np.pi/180)
isphi134.direction = 1.0

def isphi135(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 135. * (np.pi/180)
isphi135.direction = 1.0

def isphi136(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 136. * (np.pi/180)
isphi136.direction = 1.0

def isphi137(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 137. * (np.pi/180)
isphi137.direction = 1.0

def isphi138(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 138. * (np.pi/180)
isphi138.direction = 1.0

def isphi139(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 139. * (np.pi/180)
isphi139.direction = 1.0

def isphi140(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 140. * (np.pi/180)
isphi140.direction = 1.0

def isphi141(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 141. * (np.pi/180)
isphi141.direction = 1.0

def isphi142(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 142. * (np.pi/180)
isphi142.direction = 1.0

def isphi143(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 143. * (np.pi/180)
isphi143.direction = 1.0

def isphi144(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 144. * (np.pi/180)
isphi144.direction = 1.0

def isphi145(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 145. * (np.pi/180)
isphi145.direction = 1.0

def isphi146(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 146. * (np.pi/180)
isphi146.direction = 1.0

def isphi147(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 147. * (np.pi/180)
isphi147.direction = 1.0

def isphi148(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 148. * (np.pi/180)
isphi148.direction = 1.0

def isphi149(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 149. * (np.pi/180)
isphi149.direction = 1.0

def isphi150(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 150. * (np.pi/180)
isphi150.direction = 1.0

def isphi151(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 151. * (np.pi/180)
isphi151.direction = 1.0

def isphi152(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 152. * (np.pi/180)
isphi152.direction = 1.0

def isphi153(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 153. * (np.pi/180)
isphi153.direction = 1.0

def isphi154(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 154. * (np.pi/180)
isphi154.direction = 1.0

def isphi155(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 155. * (np.pi/180)
isphi155.direction = 1.0

def isphi156(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 156. * (np.pi/180)
isphi156.direction = 1.0

def isphi157(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 157. * (np.pi/180)
isphi157.direction = 1.0

def isphi158(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 158. * (np.pi/180)
isphi158.direction = 1.0

def isphi159(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 159. * (np.pi/180)
isphi159.direction = 1.0

def isphi160(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 160. * (np.pi/180)
isphi160.direction = 1.0

def isphi161(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 161. * (np.pi/180)
isphi161.direction = 1.0

def isphi162(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 162. * (np.pi/180)
isphi162.direction = 1.0

def isphi163(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 163. * (np.pi/180)
isphi163.direction = 1.0

def isphi164(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 164. * (np.pi/180)
isphi164.direction = 1.0

def isphi165(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 165. * (np.pi/180)
isphi165.direction = 1.0

def isphi166(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 166. * (np.pi/180)
isphi166.direction = 1.0

def isphi167(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 167. * (np.pi/180)
isphi167.direction = 1.0

def isphi168(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 168. * (np.pi/180)
isphi168.direction = 1.0

def isphi169(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 169. * (np.pi/180)
isphi169.direction = 1.0

def isphi170(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 170. * (np.pi/180)
isphi170.direction = 1.0

def isphi171(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 171. * (np.pi/180)
isphi171.direction = 1.0

def isphi172(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 172. * (np.pi/180)
isphi172.direction = 1.0

def isphi173(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 173. * (np.pi/180)
isphi173.direction = 1.0

def isphi174(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 174. * (np.pi/180)
isphi174.direction = 1.0

def isphi175(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 175. * (np.pi/180)
isphi175.direction = 1.0

def isphi176(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 176. * (np.pi/180)
isphi176.direction = 1.0

def isphi177(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 177. * (np.pi/180)
isphi177.direction = 1.0

def isphi178(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 178. * (np.pi/180)
isphi178.direction = 1.0

def isphi179(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 179. * (np.pi/180)
isphi179.direction = 1.0

def isphi180(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 180. * (np.pi/180)
isphi180.direction = 1.0

def isphi181(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 181. * (np.pi/180)
isphi181.direction = 1.0

def isphi182(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 182. * (np.pi/180)
isphi182.direction = 1.0

def isphi183(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 183. * (np.pi/180)
isphi183.direction = 1.0

def isphi184(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 184. * (np.pi/180)
isphi184.direction = 1.0

def isphi185(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 185. * (np.pi/180)
isphi185.direction = 1.0

def isphi186(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 186. * (np.pi/180)
isphi186.direction = 1.0

def isphi187(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 187. * (np.pi/180)
isphi187.direction = 1.0

def isphi188(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 188. * (np.pi/180)
isphi188.direction = 1.0

def isphi189(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 189. * (np.pi/180)
isphi189.direction = 1.0

def isphi190(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 190. * (np.pi/180)
isphi190.direction = 1.0

def isphi191(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 191. * (np.pi/180)
isphi191.direction = 1.0

def isphi192(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 192. * (np.pi/180)
isphi192.direction = 1.0

def isphi193(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 193. * (np.pi/180)
isphi193.direction = 1.0

def isphi194(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 194. * (np.pi/180)
isphi194.direction = 1.0

def isphi195(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 195. * (np.pi/180)
isphi195.direction = 1.0

def isphi196(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 196. * (np.pi/180)
isphi196.direction = 1.0

def isphi197(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 197. * (np.pi/180)
isphi197.direction = 1.0

def isphi198(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 198. * (np.pi/180)
isphi198.direction = 1.0

def isphi199(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 199. * (np.pi/180)
isphi199.direction = 1.0

def isphi200(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 200. * (np.pi/180)
isphi200.direction = 1.0

def isphi201(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 201. * (np.pi/180)
isphi201.direction = 1.0

def isphi202(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 202. * (np.pi/180)
isphi202.direction = 1.0

def isphi203(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 203. * (np.pi/180)
isphi203.direction = 1.0

def isphi204(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 204. * (np.pi/180)
isphi204.direction = 1.0

def isphi205(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 205. * (np.pi/180)
isphi205.direction = 1.0

def isphi206(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 206. * (np.pi/180)
isphi206.direction = 1.0

def isphi207(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 207. * (np.pi/180)
isphi207.direction = 1.0

def isphi208(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 208. * (np.pi/180)
isphi208.direction = 1.0

def isphi209(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 209. * (np.pi/180)
isphi209.direction = 1.0

def isphi210(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 210. * (np.pi/180)
isphi210.direction = 1.0

def isphi211(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 211. * (np.pi/180)
isphi211.direction = 1.0

def isphi212(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 212. * (np.pi/180)
isphi212.direction = 1.0

def isphi213(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 213. * (np.pi/180)
isphi213.direction = 1.0

def isphi214(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 214. * (np.pi/180)
isphi214.direction = 1.0

def isphi215(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 215. * (np.pi/180)
isphi215.direction = 1.0

def isphi216(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 216. * (np.pi/180)
isphi216.direction = 1.0

def isphi217(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 217. * (np.pi/180)
isphi217.direction = 1.0

def isphi218(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 218. * (np.pi/180)
isphi218.direction = 1.0

def isphi219(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 219. * (np.pi/180)
isphi219.direction = 1.0

def isphi220(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 220. * (np.pi/180)
isphi220.direction = 1.0

def isphi221(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 221. * (np.pi/180)
isphi221.direction = 1.0

def isphi222(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 222. * (np.pi/180)
isphi222.direction = 1.0

def isphi223(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 223. * (np.pi/180)
isphi223.direction = 1.0

def isphi224(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 224. * (np.pi/180)
isphi224.direction = 1.0

def isphi225(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 225. * (np.pi/180)
isphi225.direction = 1.0

def isphi226(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 226. * (np.pi/180)
isphi226.direction = 1.0

def isphi227(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 227. * (np.pi/180)
isphi227.direction = 1.0

def isphi228(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 228. * (np.pi/180)
isphi228.direction = 1.0

def isphi229(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 229. * (np.pi/180)
isphi229.direction = 1.0

def isphi230(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 230. * (np.pi/180)
isphi230.direction = 1.0

def isphi231(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 231. * (np.pi/180)
isphi231.direction = 1.0

def isphi232(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 232. * (np.pi/180)
isphi232.direction = 1.0

def isphi233(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 233. * (np.pi/180)
isphi233.direction = 1.0

def isphi234(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 234. * (np.pi/180)
isphi234.direction = 1.0

def isphi235(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 235. * (np.pi/180)
isphi235.direction = 1.0

def isphi236(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 236. * (np.pi/180)
isphi236.direction = 1.0

def isphi237(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 237. * (np.pi/180)
isphi237.direction = 1.0

def isphi238(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 238. * (np.pi/180)
isphi238.direction = 1.0

def isphi239(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 239. * (np.pi/180)
isphi239.direction = 1.0

def isphi240(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 240. * (np.pi/180)
isphi240.direction = 1.0

def isphi241(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 241. * (np.pi/180)
isphi241.direction = 1.0

def isphi242(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 242. * (np.pi/180)
isphi242.direction = 1.0

def isphi243(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 243. * (np.pi/180)
isphi243.direction = 1.0

def isphi244(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 244. * (np.pi/180)
isphi244.direction = 1.0

def isphi245(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 245. * (np.pi/180)
isphi245.direction = 1.0

def isphi246(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 246. * (np.pi/180)
isphi246.direction = 1.0

def isphi247(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 247. * (np.pi/180)
isphi247.direction = 1.0

def isphi248(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 248. * (np.pi/180)
isphi248.direction = 1.0

def isphi249(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 249. * (np.pi/180)
isphi249.direction = 1.0

def isphi250(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 250. * (np.pi/180)
isphi250.direction = 1.0

def isphi251(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 251. * (np.pi/180)
isphi251.direction = 1.0

def isphi252(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 252. * (np.pi/180)
isphi252.direction = 1.0

def isphi253(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 253. * (np.pi/180)
isphi253.direction = 1.0

def isphi254(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 254. * (np.pi/180)
isphi254.direction = 1.0

def isphi255(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 255. * (np.pi/180)
isphi255.direction = 1.0

def isphi256(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 256. * (np.pi/180)
isphi256.direction = 1.0

def isphi257(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 257. * (np.pi/180)
isphi257.direction = 1.0

def isphi258(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 258. * (np.pi/180)
isphi258.direction = 1.0

def isphi259(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 259. * (np.pi/180)
isphi259.direction = 1.0

def isphi260(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 260. * (np.pi/180)
isphi260.direction = 1.0

def isphi261(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 261. * (np.pi/180)
isphi261.direction = 1.0

def isphi262(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 262. * (np.pi/180)
isphi262.direction = 1.0

def isphi263(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 263. * (np.pi/180)
isphi263.direction = 1.0

def isphi264(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 264. * (np.pi/180)
isphi264.direction = 1.0

def isphi265(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 265. * (np.pi/180)
isphi265.direction = 1.0

def isphi266(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 266. * (np.pi/180)
isphi266.direction = 1.0

def isphi267(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 267. * (np.pi/180)
isphi267.direction = 1.0

def isphi268(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 268. * (np.pi/180)
isphi268.direction = 1.0

def isphi269(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 269. * (np.pi/180)
isphi269.direction = 1.0

def isphi270(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 270. * (np.pi/180)
isphi270.direction = 1.0

def isphi271(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 271. * (np.pi/180)
isphi271.direction = 1.0

def isphi272(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 272. * (np.pi/180)
isphi272.direction = 1.0

def isphi273(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 273. * (np.pi/180)
isphi273.direction = 1.0

def isphi274(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 274. * (np.pi/180)
isphi274.direction = 1.0

def isphi275(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 275. * (np.pi/180)
isphi275.direction = 1.0

def isphi276(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 276. * (np.pi/180)
isphi276.direction = 1.0

def isphi277(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 277. * (np.pi/180)
isphi277.direction = 1.0

def isphi278(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 278. * (np.pi/180)
isphi278.direction = 1.0

def isphi279(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 279. * (np.pi/180)
isphi279.direction = 1.0

def isphi280(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 280. * (np.pi/180)
isphi280.direction = 1.0

def isphi281(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 281. * (np.pi/180)
isphi281.direction = 1.0

def isphi282(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 282. * (np.pi/180)
isphi282.direction = 1.0

def isphi283(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 283. * (np.pi/180)
isphi283.direction = 1.0

def isphi284(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 284. * (np.pi/180)
isphi284.direction = 1.0

def isphi285(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 285. * (np.pi/180)
isphi285.direction = 1.0

def isphi286(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 286. * (np.pi/180)
isphi286.direction = 1.0

def isphi287(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 287. * (np.pi/180)
isphi287.direction = 1.0

def isphi288(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 288. * (np.pi/180)
isphi288.direction = 1.0

def isphi289(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 289. * (np.pi/180)
isphi289.direction = 1.0

def isphi290(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 290. * (np.pi/180)
isphi290.direction = 1.0

def isphi291(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 291. * (np.pi/180)
isphi291.direction = 1.0

def isphi292(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 292. * (np.pi/180)
isphi292.direction = 1.0

def isphi293(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 293. * (np.pi/180)
isphi293.direction = 1.0

def isphi294(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 294. * (np.pi/180)
isphi294.direction = 1.0

def isphi295(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 295. * (np.pi/180)
isphi295.direction = 1.0

def isphi296(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 296. * (np.pi/180)
isphi296.direction = 1.0

def isphi297(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 297. * (np.pi/180)
isphi297.direction = 1.0

def isphi298(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 298. * (np.pi/180)
isphi298.direction = 1.0

def isphi299(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 299. * (np.pi/180)
isphi299.direction = 1.0

def isphi300(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 300. * (np.pi/180)
isphi300.direction = 1.0

def isphi301(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 301. * (np.pi/180)
isphi301.direction = 1.0

def isphi302(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 302. * (np.pi/180)
isphi302.direction = 1.0

def isphi303(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 303. * (np.pi/180)
isphi303.direction = 1.0

def isphi304(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 304. * (np.pi/180)
isphi304.direction = 1.0

def isphi305(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 305. * (np.pi/180)
isphi305.direction = 1.0

def isphi306(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 306. * (np.pi/180)
isphi306.direction = 1.0

def isphi307(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 307. * (np.pi/180)
isphi307.direction = 1.0

def isphi308(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 308. * (np.pi/180)
isphi308.direction = 1.0

def isphi309(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 309. * (np.pi/180)
isphi309.direction = 1.0

def isphi310(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 310. * (np.pi/180)
isphi310.direction = 1.0

def isphi311(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 311. * (np.pi/180)
isphi311.direction = 1.0

def isphi312(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 312. * (np.pi/180)
isphi312.direction = 1.0

def isphi313(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 313. * (np.pi/180)
isphi313.direction = 1.0

def isphi314(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 314. * (np.pi/180)
isphi314.direction = 1.0

def isphi315(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 315. * (np.pi/180)
isphi315.direction = 1.0

def isphi316(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 316. * (np.pi/180)
isphi316.direction = 1.0

def isphi317(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 317. * (np.pi/180)
isphi317.direction = 1.0

def isphi318(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1],p_XYZ[0])
	if phi<0.: phi += 2*np.pi
	return phi - 318. * (np.pi/180)
isphi318.direction = 1.0

def isphi319(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 319. * (np.pi / 180)
isphi319.direction = 1.0

def isphi320(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 320. * (np.pi / 180)
isphi320.direction = 1.0

def isphi321(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 321. * (np.pi / 180)
isphi321.direction = 1.0

def isphi322(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 322. * (np.pi / 180)
isphi322.direction = 1.0

def isphi323(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 323. * (np.pi / 180)
isphi323.direction = 1.0

def isphi324(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 324. * (np.pi / 180)
isphi324.direction = 1.0

def isphi325(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 325. * (np.pi / 180)
isphi325.direction = 1.0

def isphi326(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 326. * (np.pi / 180)
isphi326.direction = 1.0

def isphi327(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 327. * (np.pi / 180)
isphi327.direction = 1.0

def isphi328(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 328. * (np.pi / 180)
isphi328.direction = 1.0

def isphi329(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 329. * (np.pi / 180)
isphi329.direction = 1.0

def isphi330(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 330. * (np.pi / 180)
isphi330.direction = 1.0

def isphi331(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 331. * (np.pi / 180)
isphi331.direction = 1.0

def isphi332(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 332. * (np.pi / 180)
isphi332.direction = 1.0

def isphi333(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 333. * (np.pi / 180)
isphi333.direction = 1.0

def isphi334(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 334. * (np.pi / 180)
isphi334.direction = 1.0

def isphi335(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 335. * (np.pi / 180)
isphi335.direction = 1.0

def isphi336(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 336. * (np.pi / 180)
isphi336.direction = 1.0

def isphi337(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 337. * (np.pi / 180)
isphi337.direction = 1.0

def isphi338(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 338. * (np.pi / 180)
isphi338.direction = 1.0

def isphi339(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 339. * (np.pi / 180)
isphi339.direction = 1.0

def isphi340(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 340. * (np.pi / 180)
isphi340.direction = 1.0

def isphi341(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 341. * (np.pi / 180)
isphi341.direction = 1.0

def isphi342(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 342. * (np.pi / 180)
isphi342.direction = 1.0

def isphi343(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 343. * (np.pi / 180)
isphi343.direction = 1.0

def isphi344(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 344. * (np.pi / 180)
isphi344.direction = 1.0

def isphi345(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 345. * (np.pi / 180)
isphi345.direction = 1.0

def isphi346(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 346. * (np.pi / 180)
isphi346.direction = 1.0

def isphi347(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 347. * (np.pi / 180)
isphi347.direction = 1.0

def isphi348(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 348. * (np.pi / 180)
isphi348.direction = 1.0

def isphi349(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 349. * (np.pi / 180)
isphi349.direction = 1.0

def isphi350(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 350. * (np.pi / 180)
isphi350.direction = 1.0

def isphi351(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 351. * (np.pi / 180)
isphi351.direction = 1.0

def isphi352(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 352. * (np.pi / 180)
isphi352.direction = 1.0

def isphi353(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 353. * (np.pi / 180)
isphi353.direction = 1.0

def isphi354(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 354. * (np.pi / 180)
isphi354.direction = 1.0

def isphi355(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 355. * (np.pi / 180)
isphi355.direction = 1.0

def isphi356(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 356. * (np.pi / 180)
isphi356.direction = 1.0

def isphi357(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 357. * (np.pi / 180)
isphi357.direction = 1.0

def isphi358(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 358. * (np.pi / 180)
isphi358.direction = 1.0

def isphi359(t, p_XYZ, Mesh):
	phi = (-1) * np.arctan2(p_XYZ[1], p_XYZ[0])
	if phi < 0.: phi += 2 * np.pi
	return phi - 359. * (np.pi / 180)
isphi359.direction = 1.0

def isphi360(t, p_XYZ, Mesh):
	return p_XYZ[1]
isphi360.direction = -1.0
