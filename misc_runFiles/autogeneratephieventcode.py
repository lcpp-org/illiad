##This file rewrites the phi_events.py file with the different functions corresponding to the spacing specified by eventspacing
##It creates a function for every event taking place in the 360 degrees
##It also creates a list called poincare events which has a list of all of the functions that have been created in the this file
##Finally it specifies the phi range by creating an array with all of the degrees events will occur at

def create_phi_events(name, eventspacing):
    #These lines are required for every spacing, they setup the imports and define the first function
    initial_lines = ["import numpy as np\n", "import numba as nb\n", "from coordtrans import XYZ_to_RTP\n", "from mesh import *\n", "import {} \n".format(name[:-3]), \
                "#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, \"C\"), Mesh.class_type.instance_type), nopython=True)\n", \
                "def inVV(t, p_XYZ, Mesh):\n", "	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)\n", "	return Mesh.a - p_RTP[0]\n", \
                "inVV.direction = -1.0\n", "inVV.terminal = True\n\n\n\n\n"]
    ##poincare_gen_lines is where the list of events will be stored before being written to the file
    poincare_gen_lines = ["    poincare_events = [ {}.inVV, \n".format(name[:-3])]

    if 360%eventspacing != 0:
        return("error: bad spacing")
    else:
        numEvents = 360 // eventspacing
        with open(name, "w") as f:
            f.writelines(initial_lines)
        
        for i in range(0,numEvents): ##for every event in a full rotation
            degree = (i+1)*eventspacing
            ##creates a function at that degree angle with the specifics associated with the required direction 
            # (1 means particles crossing it low_phi to high_phi, -1 is in the other direction and 0 is either direction)
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
            #writes all of the functions in     
            with open(name, "a") as f:
                f.writelines(event_lines)

        with open(name, "a") as f:
            #the poincare_events list and phi_range have to be accessible: this function is how they are made accessible in other scripts
            f.write("def eventsAndRange(): \n")
            f.writelines(poincare_gen_lines)
            f.write("    phi_range = np.linspace(np.pi/{}., 2*np.pi, {}) \n".format(int(180/eventspacing), int(360/eventspacing)))
            f.write("    return poincare_events,phi_range \n")

create_phi_events("phi_events.py", 9)