
initial_lines = ["import numpy as np\n", "import numba as nb\n", "from coordtrans import XYZ_to_RTP\n", "from mesh import *\n", \
                "#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, \"C\"), Mesh.class_type.instance_type), nopython=True)\n", \
                "def inVV(t, p_XYZ, Mesh):\n", "	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)\n", "	return Mesh.a - p_RTP[0]\n", \
                "inVV.direction = -1.0\n", "inVV.terminal = True\n\n\n\n\n"]



def create_phi_events(name, name2, eventspacing):
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
            else: 
                direction = -1.0
                poincare_gen_lines.append(f"                        {name[:-3]}.isphi{degree}] \n \n \n")
            
            event_lines = ["#@nb.jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, \"C\"), Mesh.class_type.instance_type), nopython=True)\n", 
                          f"def isphi{degree}(t, p_XYZ, Mesh):\n", "	p_RTP = XYZ_to_RTP(p_XYZ[:3], Mesh.R0)\n", "	return p_RTP[2] - 1. * (np.pi/20.)\n",
                          f"isphi{degree}.direction = {direction}\n\n"]
            with open(name, "a") as f:
                f.writelines(event_lines)
    
    with open(name2, "r+") as f:
        i = 0
        
        contents = f.readlines()
        
        start = 0
        stop = 0
        for line in contents:
            if "## EVENT LIST" in line:
                i+=1
                start = i
            elif "## SOLVER" in line and start != 0:
                stop = i
                break
            else:
                i+=1
        
        new_contents = contents[:start]
        
        for line in contents[stop:]:
            new_contents.append(line)
        
        for j in range(len(poincare_gen_lines)):
            new_contents.insert(start+j, poincare_gen_lines[j])
        
        new_contents[6] = f"import {name[:-3]} \n"
        done = 0
        for i,line in enumerate(new_contents):
            if "phi_range = np.linspace( np.pi/20., 2*np.pi," in line:
                new_contents[i] = f"    phi_range = np.linspace( np.pi/20., 2*np.pi, {360/eventspacing:.0f}) \n"
                done = 1
                
            elif "with cf.ProcessPoolExecutor(max_workers=" in line and done == 1:
                new_contents[i] = f"    with cf.ProcessPoolExecutor(max_workers={360/eventspacing:.0f}) as executor: \n"
                


        f.seek(0)
        f.truncate()
        f.writelines(new_contents)
    
create_phi_events("phi_events_5.py","variable_poincare_gen.py", 5)