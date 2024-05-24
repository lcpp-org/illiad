import numpy as np
import numba as nb

from coordtrans import XYZ_to_RTP
from mesh import *

def inVV(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return Mesh.a - p_RTP[0]
inVV.direction = -1.0
inVV.terminal = True


@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi9(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 1. * (np.pi/20.) 
isphi9.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi18(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 2. * (np.pi/20.)
isphi18.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi27(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 3. * (np.pi/20.)
isphi27.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi36(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 4. * (np.pi/20.)
isphi36.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi45(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 5. * (np.pi/20.)
isphi45.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi54(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 6. * (np.pi/20.)
isphi54.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi63(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 7. * (np.pi/20.) 
isphi63.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi72(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 8. * (np.pi/20.)
isphi72.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi81(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 9. * (np.pi/20.)
isphi81.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi90(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 10. * (np.pi/20.)
isphi90.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi99(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 11. * (np.pi/20.)
isphi99.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi108(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 12. * (np.pi/20.)
isphi108.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi117(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 13. * (np.pi/20.)
isphi117.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi126(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 14. * (np.pi/20.)
isphi126.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi135(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 15. * (np.pi/20.)
isphi135.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi144(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 16. * (np.pi/20.)
isphi144.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi153(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 17. * (np.pi/20.)
isphi153.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi162(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 18. * (np.pi/20.)
isphi162.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi171(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 19. * (np.pi/20.)
isphi171.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi180(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 20. * (np.pi/20.)
isphi180.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi189(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 21. * (np.pi/20.)
isphi189.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi198(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 22. * (np.pi/20.)
isphi198.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi207(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 23. * (np.pi/20.)
isphi207.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi216(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 24. * (np.pi/20.)
isphi216.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi225(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 25. * (np.pi/20.)
isphi225.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi234(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 26. * (np.pi/20.)
isphi234.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi243(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 27. * (np.pi/20.)
isphi243.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi252(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 28. * (np.pi/20.)
isphi252.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi261(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 29. * (np.pi/20.)
isphi261.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi270(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 30. * (np.pi/20.)
isphi270.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi279(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 31. * (np.pi/20.)
isphi279.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi288(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 32. * (np.pi/20.)
isphi288.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi297(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 33. * (np.pi/20.)
isphi297.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi306(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 34. * (np.pi/20.)
isphi306.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi315(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 35. * (np.pi/20.)
isphi315.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi324(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 36. * (np.pi/20.)
isphi324.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi333(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 37. * (np.pi/20.)
isphi333.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi342(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 38. * (np.pi/20.)
isphi342.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi351(t, p_XYZ, Mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)
	return p_RTP[2] - 39. * (np.pi/20.)
isphi351.direction = 1.0

@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C"), Mesh.class_type.instance_type), nopython=True)
def isphi360(t, p_XYZ, Mesh):
	return p_XYZ[1]
isphi360.direction = -1