




def create_phi_events(name, eventspacing):
    initial_lines = ["import numpy as np\n", "import numba as nb\n", "from coordtrans import XYZ_to_RTP\n", "from mesh import *\n", f"import {name[:-3]} \n", \
                "#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, \"C\"), Mesh.class_type.instance_type), nopython=True)\n", \
                "def inVV(t, p_XYZ, Mesh):\n", "	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)\n", "	return Mesh.a - p_RTP[0]\n", \
                "inVV.direction = -1.0\n", "inVV.terminal = True\n\n\n\n\n"]
    
    poincare_gen_lines = [f"    poincare_events = [ {name[:-3]}.inVV, \n"]

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
                poincare_gen_lines.append(f"                        {name[:-3]}.isphi{degree}, \n")
                event_lines = ["#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, \"C\"), Mesh.class_type.instance_type), nopython=True)\n", 
                          f"def isphi{degree}(t, p_XYZ, Mesh):\n", "	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)\n", f"	return p_RTP[2] - {i+1}. * (np.pi/{numEvents//2})\n",
                          f"isphi{degree}.direction = {direction}\n\n"]
            else: 
                direction = -1.0
                poincare_gen_lines.append(f"                        {name[:-3]}.isphi{degree}] \n")
            
                event_lines = ["#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, \"C\"), Mesh.class_type.instance_type), nopython=True)\n", 
                            f"def isphi{degree}(t, p_XYZ, Mesh):\n", f"	return p_XYZ[1]\n",
                            f"isphi{degree}.direction = {direction}\n\n"]
                
            with open(name, "a") as f:
                f.writelines(event_lines)

        with open(name, "a") as f:
            f.write("def eventsAndRange(): \n")
            f.writelines(poincare_gen_lines)
            f.write(f"    phi_range = np.linspace(np.pi/{180/eventspacing:.0f}., 2*np.pi, {360/eventspacing:.0f}) \n")
            f.write("    return poincare_events,phi_range \n")

create_phi_events("phi_events.py", 1)