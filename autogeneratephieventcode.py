initial_lines = ["import numpy as np\n", "import numba as nb\n", "from coordtrans import XYZ_to_RTP\n", "from mesh import *\n", \
                "#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, \"C\"), Mesh.class_type.instance_type), nopython=True)\n", \
                "def inVV(t, p_XYZ, Mesh):\n", "	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)\n", "	return Mesh.a - p_RTP[0]\n", \
                "inVV.direction = -1.0\n", "inVV.terminal = True\n\n\n\n\n"]

def create_phi_events(name, eventspacing):
    if 360%eventspacing != 0:
        return("error: bad spacing")
    else:
        numEvents = 360 // eventspacing
        with open(name, "w") as f:
            f.writelines(initial_lines)
        
        for i in range(0,numEvents):
            degree = (i+1)*eventspacing
            print(degree, eventspacing, i)
            if degree != 360:
                direction = 1.0
            else: 
                direction = -1.0
            event_lines = ["#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, \"C\"), Mesh.class_type.instance_type), nopython=True)\n", 
                          f"def isphi{degree}(t, p_XYZ, Mesh):\n", "	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)\n", "	return p_RTP[2] - 1. * (np.pi/20.)\n",
                          f"isphi{degree}.direction = {direction}\n\n"]
            with open(name, "a") as f:
                f.writelines(event_lines)


create_phi_events("phi_events_3.py", 3)