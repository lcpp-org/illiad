import numpy as np
#from math import degrees
from functools import partial
import concurrent.futures as cf
from time import perf_counter

#import phi_events
from phi_events import *
from utility.anlys_funcs import Output_Poincare
from solver.ode import solvePoincare
from utility.coordtrans import XYZ_to_RTP, RTP_to_XYZ
from classes.particle import *

#def Gen_Poincare(field_, fieldlines, outputHandler, anlys_name, solvr, rtl_, atl_, workers=40, saveData=True):
def Gen_Poincare(ic_rtp_arr, spins, field_, outputHandler, anlys_name, solvr='LSODA', rtl_=1e-6, atl_=1e-16, workers=6, double_line=False, saveData=True):
    outputHandler.createSubDir(anlys_name)
    
    ## CONVERT TO XYZ COORDS
    NLINES = len(ic_rtp_arr)
    ICs_XYZ = np.zeros(shape=(NLINES, 3))
    for i in range(NLINES):
        ICs_XYZ[i] = RTP_to_XYZ(ic_rtp_arr[i], field_.R0)

    # Print out a nicely-formatted boilerplate listing the parameters and their values as a table with a border
    outputHandler.log.info("+----------------+-------------------------+")
    outputHandler.log.info("| Parameter      | Value                   |")
    outputHandler.log.info("+----------------+-------------------------+")
    outputHandler.log.info(f"| SOLVER         | {solvr:<23} |")
    outputHandler.log.info(f"| RTOL           | {rtl_:<23} |")
    outputHandler.log.info(f"| ATOL           | {atl_:<23} |")
    outputHandler.log.info(f"| THREADS        | {workers:<23} |")
    outputHandler.log.info("+----------------+-------------------------+")
    outputHandler.log.info(f"| NLINES         | {NLINES:<23} |")
    outputHandler.log.info(f"| SPINS          | {spins:<23} |")
    outputHandler.log.info("| Initial Conditions (RTP):                |")
    for ic in ic_rtp_arr:
        outputHandler.log.info(f"|     {str(ic):<23}   |")
    outputHandler.log.info("+----------------+-------------------------+")


    ## GENERATE POINCARE DATA
    length = (2*np.pi * field_.R0) * spins
    fieldlines = [fieldLine(init_cond, length, direction = 1.0) for init_cond in ICs_XYZ]

    ## SOLVER SETUP
    Nlines = Particle.particleCount

    if double_line:
        # Add fieldlines in opposite direction
        fieldlines += [fieldLine(init_cond, length, direction = -1.0) for init_cond in ICs_XYZ]

    ## EVENT LIST (avert your eyes)
    """
    poincare_events = [ inVV,
                        isphi9,
                        isphi18,
                        isphi27,
                        isphi36,
                        isphi45,
                        isphi54,
                        isphi63,
                        isphi72,
                        isphi81,
                        isphi90,
                        isphi99,
                        isphi108,
                        isphi117,
                        isphi126,
                        isphi135,
                        isphi144,
                        isphi153,
                        isphi162,
                        isphi171,
                        isphi180,
                        isphi189,
                        isphi198,
                        isphi207,
                        isphi216,
                        isphi225,
                        isphi234,
                        isphi243,
                        isphi252,
                        isphi261,
                        isphi270,
                        isphi279,
                        isphi288,
                        isphi297,
                        isphi306,
                        isphi315,
                        isphi324,
                        isphi333,
                        isphi342,
                        isphi351,
                        isphi360]
    """

    poincare_events = [ inVV, 
                        isphi1, 
                        isphi2, 
                        isphi3, 
                        isphi4, 
                        isphi5, 
                        isphi6, 
                        isphi7, 
                        isphi8, 
                        isphi9, 
                        isphi10, 
                        isphi11, 
                        isphi12, 
                        isphi13, 
                        isphi14, 
                        isphi15, 
                        isphi16, 
                        isphi17, 
                        isphi18, 
                        isphi19, 
                        isphi20, 
                        isphi21, 
                        isphi22, 
                        isphi23, 
                        isphi24, 
                        isphi25, 
                        isphi26, 
                        isphi27, 
                        isphi28, 
                        isphi29, 
                        isphi30, 
                        isphi31, 
                        isphi32, 
                        isphi33, 
                        isphi34, 
                        isphi35, 
                        isphi36, 
                        isphi37, 
                        isphi38, 
                        isphi39, 
                        isphi40, 
                        isphi41, 
                        isphi42, 
                        isphi43, 
                        isphi44, 
                        isphi45, 
                        isphi46, 
                        isphi47, 
                        isphi48, 
                        isphi49, 
                        isphi50, 
                        isphi51, 
                        isphi52, 
                        isphi53, 
                        isphi54, 
                        isphi55, 
                        isphi56, 
                        isphi57, 
                        isphi58, 
                        isphi59, 
                        isphi60, 
                        isphi61, 
                        isphi62, 
                        isphi63, 
                        isphi64, 
                        isphi65, 
                        isphi66, 
                        isphi67, 
                        isphi68, 
                        isphi69, 
                        isphi70, 
                        isphi71, 
                        isphi72, 
                        isphi73, 
                        isphi74, 
                        isphi75, 
                        isphi76, 
                        isphi77, 
                        isphi78, 
                        isphi79, 
                        isphi80, 
                        isphi81, 
                        isphi82, 
                        isphi83, 
                        isphi84, 
                        isphi85, 
                        isphi86, 
                        isphi87, 
                        isphi88, 
                        isphi89, 
                        isphi90, 
                        isphi91, 
                        isphi92, 
                        isphi93, 
                        isphi94, 
                        isphi95, 
                        isphi96, 
                        isphi97, 
                        isphi98, 
                        isphi99, 
                        isphi100, 
                        isphi101, 
                        isphi102, 
                        isphi103, 
                        isphi104, 
                        isphi105, 
                        isphi106, 
                        isphi107, 
                        isphi108, 
                        isphi109, 
                        isphi110, 
                        isphi111, 
                        isphi112, 
                        isphi113, 
                        isphi114, 
                        isphi115, 
                        isphi116, 
                        isphi117, 
                        isphi118, 
                        isphi119, 
                        isphi120, 
                        isphi121, 
                        isphi122, 
                        isphi123, 
                        isphi124, 
                        isphi125, 
                        isphi126, 
                        isphi127, 
                        isphi128, 
                        isphi129, 
                        isphi130, 
                        isphi131, 
                        isphi132, 
                        isphi133, 
                        isphi134, 
                        isphi135, 
                        isphi136, 
                        isphi137, 
                        isphi138, 
                        isphi139, 
                        isphi140, 
                        isphi141, 
                        isphi142, 
                        isphi143, 
                        isphi144, 
                        isphi145, 
                        isphi146, 
                        isphi147, 
                        isphi148, 
                        isphi149, 
                        isphi150, 
                        isphi151, 
                        isphi152, 
                        isphi153, 
                        isphi154, 
                        isphi155, 
                        isphi156, 
                        isphi157, 
                        isphi158, 
                        isphi159, 
                        isphi160, 
                        isphi161, 
                        isphi162, 
                        isphi163, 
                        isphi164, 
                        isphi165, 
                        isphi166, 
                        isphi167, 
                        isphi168, 
                        isphi169, 
                        isphi170, 
                        isphi171, 
                        isphi172, 
                        isphi173, 
                        isphi174, 
                        isphi175, 
                        isphi176, 
                        isphi177, 
                        isphi178, 
                        isphi179, 
                        isphi180, 
                        isphi181, 
                        isphi182, 
                        isphi183, 
                        isphi184, 
                        isphi185, 
                        isphi186, 
                        isphi187, 
                        isphi188, 
                        isphi189, 
                        isphi190, 
                        isphi191, 
                        isphi192, 
                        isphi193, 
                        isphi194, 
                        isphi195, 
                        isphi196, 
                        isphi197, 
                        isphi198, 
                        isphi199, 
                        isphi200, 
                        isphi201, 
                        isphi202, 
                        isphi203, 
                        isphi204, 
                        isphi205, 
                        isphi206, 
                        isphi207, 
                        isphi208, 
                        isphi209, 
                        isphi210, 
                        isphi211, 
                        isphi212, 
                        isphi213, 
                        isphi214, 
                        isphi215, 
                        isphi216, 
                        isphi217, 
                        isphi218, 
                        isphi219, 
                        isphi220, 
                        isphi221, 
                        isphi222, 
                        isphi223, 
                        isphi224, 
                        isphi225, 
                        isphi226, 
                        isphi227, 
                        isphi228, 
                        isphi229, 
                        isphi230, 
                        isphi231, 
                        isphi232, 
                        isphi233, 
                        isphi234, 
                        isphi235, 
                        isphi236, 
                        isphi237, 
                        isphi238, 
                        isphi239, 
                        isphi240, 
                        isphi241, 
                        isphi242, 
                        isphi243, 
                        isphi244, 
                        isphi245, 
                        isphi246, 
                        isphi247, 
                        isphi248, 
                        isphi249, 
                        isphi250, 
                        isphi251, 
                        isphi252, 
                        isphi253, 
                        isphi254, 
                        isphi255, 
                        isphi256, 
                        isphi257, 
                        isphi258, 
                        isphi259, 
                        isphi260, 
                        isphi261, 
                        isphi262, 
                        isphi263, 
                        isphi264, 
                        isphi265, 
                        isphi266, 
                        isphi267, 
                        isphi268, 
                        isphi269, 
                        isphi270, 
                        isphi271, 
                        isphi272, 
                        isphi273, 
                        isphi274, 
                        isphi275, 
                        isphi276, 
                        isphi277, 
                        isphi278, 
                        isphi279, 
                        isphi280, 
                        isphi281, 
                        isphi282, 
                        isphi283, 
                        isphi284, 
                        isphi285, 
                        isphi286, 
                        isphi287, 
                        isphi288, 
                        isphi289, 
                        isphi290, 
                        isphi291, 
                        isphi292, 
                        isphi293, 
                        isphi294, 
                        isphi295, 
                        isphi296, 
                        isphi297, 
                        isphi298, 
                        isphi299, 
                        isphi300, 
                        isphi301, 
                        isphi302, 
                        isphi303, 
                        isphi304, 
                        isphi305, 
                        isphi306, 
                        isphi307, 
                        isphi308, 
                        isphi309, 
                        isphi310, 
                        isphi311, 
                        isphi312, 
                        isphi313, 
                        isphi314, 
                        isphi315, 
                        isphi316, 
                        isphi317, 
                        isphi318, 
                        isphi319, 
                        isphi320, 
                        isphi321, 
                        isphi322, 
                        isphi323, 
                        isphi324, 
                        isphi325, 
                        isphi326, 
                        isphi327, 
                        isphi328, 
                        isphi329, 
                        isphi330, 
                        isphi331, 
                        isphi332, 
                        isphi333, 
                        isphi334, 
                        isphi335, 
                        isphi336, 
                        isphi337, 
                        isphi338, 
                        isphi339, 
                        isphi340, 
                        isphi341, 
                        isphi342, 
                        isphi343, 
                        isphi344, 
                        isphi345, 
                        isphi346, 
                        isphi347, 
                        isphi348, 
                        isphi349, 
                        isphi350, 
                        isphi351, 
                        isphi352, 
                        isphi353, 
                        isphi354, 
                        isphi355, 
                        isphi356, 
                        isphi357, 
                        isphi358, 
                        isphi359, 
                        isphi360] 

    ## SOLVER
    length = fieldlines[0].maxLife
    solvePoincare_x = partial(solvePoincare, field=field_, solver=solvr, rtl= rtl_, atl=atl_, solver_events=poincare_events)

    ## PARALLELIZATION WITH CONCURRENT FUTURES 'MAP' OVER EACH PARTICLE
    outputHandler.log.info('Begin running {} ICs for max. {} spins...'.format(Nlines, int(length/(2*np.pi * field_.R0))))
    t_start = perf_counter()
    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        solver_output = executor.map(solvePoincare_x, fieldlines)
    t_stop = perf_counter()
    tot_elapsed_time = t_stop - t_start
    outputHandler.log.info('ALL SOLVERS FINISHED IN {} seconds\n###############\n\n'.format(tot_elapsed_time))

    ## PARSE OUTPUT INTO LISTS
    pathLength_=[]
    Poincare_output_ = []
    wall_output_ = []

    for pLngth, out in solver_output:
        pathLength_ += [pLngth]
        Poincare_output_ += [out[1:]]
        if out[0].any():
            wall_output_ += [XYZ_to_RTP(out[0][0], field_.R0)]

    if double_line:
        # Combine the positive and negative fieldlines into one
        pathLength_ = [pathLength_[i]+pathLength_[i+NLINES] for i in range(0,NLINES)]
        for line_index in range(0,NLINES):
            for event_index in range(len(Poincare_output_[line_index])):
                arr_a = Poincare_output_[line_index][event_index]
                arr_b = Poincare_output_[line_index+NLINES][event_index]
                if arr_a.any() and arr_b.any():
                    Poincare_output_[line_index][event_index] = np.vstack((arr_a, arr_b))
        Poincare_output_ = Poincare_output_[:NLINES]


    ## POST-SOLVER OUTPUT
    ####################
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'figure.autolayout':True})

    outputHandler.log.info('PLOTTING AND OUTPUTTING PHI-ANGLE DATA:')
    #phi_range = np.linspace( np.pi/20., 2*np.pi, 40)
    phi_range = np.linspace( np.pi/180., 2*np.pi, 360)

    plot_workers = min(workers, 40)
    # LOOPING OVER EACH PHI ANGLE
    iter_in = enumerate(phi_range)
    Output_Poincare_x = partial(Output_Poincare, field_=field_, Pdata=Poincare_output_,
                                 anlys_name=anlys_name, outputHandler=outputHandler, saveData=saveData)
    with cf.ProcessPoolExecutor(max_workers=plot_workers) as executor:
        outs = executor.map(Output_Poincare_x, iter_in)
    
    # EXECUTE GENERATOR
    for out in outs:
        outputHandler.log.info(out)


    return pathLength_, Poincare_output_, wall_output_