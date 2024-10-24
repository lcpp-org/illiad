




def create_phi_events(name, eventspacing):
    initial_lines = ["import numpy as np\n", "import numba as nb\n", "from coordtrans import XYZ_to_RTP\n", "from mesh import *\n", "import {} \n".format(name[:-3]), \
                "#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, \"C\"), Mesh.class_type.instance_type), nopython=True)\n", \
                "def inVV(t, p_XYZ, Mesh):\n", "	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)\n", "	return Mesh.a - p_RTP[0]\n", \
                "inVV.direction = -1.0\n", "inVV.terminal = True\n\n\n\n\n"]
    
    poincare_gen_lines = ["    poincare_events = [ {}.inVV, \n".format(name[:-3])]

    if 360%eventspacing != 0:
        return("error: bad spacing")
    else:
        numEvents = 360 // eventspacing
        with open(name, "w") as f:
            f.writelines(initial_lines)
        
        for i in range(0,numEvents):
            degree = (i+1)*eventspacing
            
            if degree != 360:
                direction = 1.0
                poincare_gen_lines.append("                        {}.isphi{}, \n".format(name[:-3], degree))
                event_lines = ["#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, \"C\"), Mesh.class_type.instance_type), nopython=True)\n", 
                          "def isphi{}(t, p_XYZ, Mesh):\n".format(degree), "	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)\n", "	return p_RTP[2] - {}. * (np.pi/{})\n".format(i+1, numEvents//2),
                          "isphi{}.direction = {}\n\n".format(degree, direction)]
            else: 
                direction = -1.0
                poincare_gen_lines.append("                        {}.isphi{}] \n".format(name[:-3], degree))
            
                event_lines = ["#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, \"C\"), Mesh.class_type.instance_type), nopython=True)\n", 
                            "def isphi{}(t, p_XYZ, Mesh):\n".format(degree), "	return p_XYZ[1]\n",
                            "isphi{}.direction = {}\n\n".format(degree, direction)]
                
            with open(name, "a") as f:
                f.writelines(event_lines)

        with open(name, "a") as f:
            f.write("def eventsAndRange(): \n")
            f.writelines(poincare_gen_lines)
            f.write("    phi_range = np.linspace(np.pi/{}., 2*np.pi, {}) \n".format(int(180/eventspacing), int(360/eventspacing)))
            f.write("    return poincare_events,phi_range \n")

create_phi_events("phi_events.py", 1)