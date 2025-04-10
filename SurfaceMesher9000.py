import numpy as np
import matplotlib.pyplot as plt
import classes.class_outputHandler as out
from utility.coordtrans import RTP_to_XYZ
import gmsh

##################################################
# NOT SHIFTED IN PHI (computational to physical) #
##################################################
def main():

    # INITIALIZE IO HANDLER
    simIO = out.IOHandler(ANLYS_DIR)
    simIO.startLog()
    simIO.createSubDir(ANLYS_SUBDIR)

    # INITIALIZE GMSH!
    gmsh.initialize() #["-nt", "32"])
    if gmsh.model.getCurrent(): gmsh.clear()
    gmsh.model.add("Flux_Surf_Mesh")
    fluxMesh = gmsh.model.occ

    # LOAD THE FLUX SURFACE
    filename_pts = 'fSurf_{:03d}_POINTmesh.npy'.format(SURFACE_INDX)
    flux_surface = simIO.loadNumpyData(filename_pts, subdir=ANLYS_SUBDIR)

    # LOAD THE ISLAND CENTERS
    filename_cntrs = 'fSurf_{:03d}_center.npy'.format(SURFACE_INDX)
    center_point = simIO.loadNumpyData(filename_pts, subdir=ANLYS_SUBDIR)


    # MAX_NUM_SUBSETS = 3 #!!!!  explicitly set to # of subsets in file
    # NPHI_MESH = flux_surface.shape[0]
    # NTHETA_MESH = flux_surface.shape[1] // MAX_NUM_SUBSETS

    # CALCULATE THE STRIDE FOR THE SUBSETS
    PHISTRIDE, phi_rmdr = np.divmod(NPHI_MESH, NPHI)
    THETASTRIDE, thet_rmdr = np.divmod(NTHETA_MESH, NTHETA)
    assert phi_rmdr == 0, 'NPHI_MESH is not divisible by NPHI'
    assert thet_rmdr == 0, 'NTHETA_MESH is not divisible by NTHETA'
    phi_arr_deg = np.linspace(360./NPHI, 360., NPHI)# Define the phi angles to CREATE LOOPS

    # SET OUTPUT NAMES
    meshString = str(NUM_SUBSETS)+'x'+str(NPHI)+'x'+str(NTHETA)
    gmsh_filename = 'HIDRA_STRIDING-5_' + meshString + '_' + str(SURFACE_INDX)# +'_3rd12'


    # PHI LOOP
    curveLoop_tag_list = []
    curveLoop_tag_list_2 = []
    curveLoop_tag_list_3 = []
    phi_comp_to_phys = 198.
    simIO.log.info('STARTING PHI LOOP!')
    for phi_index, phi in enumerate(phi_arr_deg):

        # TRANSLATE THE BETWEEN MESH PHI INDEX AND DESIRED PHI INDEX
        phi_index = PHISTRIDE * phi_index + PHISTRIDE - 1

        # SLICE THE FLUX SURFACE TO GET DESIRED THETAS AND PHIS
        full_surface = flux_surface[phi_index][::THETASTRIDE]

        #this_center = center_point[phi_index]
        #simIO.log.info(f'Center point: {this_center}')

        # Shift the indices relative to their chunks
        all_shifted_ind = np.zeros([NUM_SUBSETS, NTHETA], dtype=int)
        for subset_index in range(NUM_SUBSETS):
            all_shifted_ind[subset_index] = NTHETA*subset_index + np.arange(NTHETA)

        # GATHER THE (RTP)'s, CONVERT TO (XYZ)'s
        thetas, rads = full_surface.T
        assert np.count_nonzero(~np.isnan(rads)) > 0, 'No non-Nan values in rads!'
        #simIO.log.info(f'Phi: {phi}, {thetas=}')
        phis = np.full_like(rads, (phi+phi_comp_to_phys)*np.pi/180)
        xs, ys, zs = RTP_to_XYZ(np.array([rads, thetas, phis]))

        #xs *= 100.
        #ys *= 100.
        #zs *= 100.

        # Find the index of the first subset
        index_setA = set_closest_to_zero(thetas, all_shifted_ind, NUM_SUBSETS)

        # Plots used to debug the island grouping, labelling, start index, etc
        if DEBUG_PLOT: output_debug_fig(phi, thetas, rads, all_shifted_ind, index_setA, simIO)

        # Lists of point tags for each subset (of 3)
        point_tag_list = []
        point_tag_list_2 = []
        point_tag_list_3 = []

        ## JUST np.roll() all_shifted_ind across 1st axis!
        ####################################
        # THETA LOOP(S)
        if NUM_SUBSETS == 3:
            for i in range(NTHETA):
                shifted_i      = all_shifted_ind[0][i]
                next_shifted_i = all_shifted_ind[1][i]
                last_shifted_i = all_shifted_ind[2][i]

                if index_setA == 0:
                    point_tag_list.append(fluxMesh.add_point(xs[shifted_i], ys[shifted_i], zs[shifted_i]))
                    point_tag_list_2.append(fluxMesh.add_point(xs[next_shifted_i], ys[next_shifted_i], zs[next_shifted_i]))
                    point_tag_list_3.append(fluxMesh.add_point(xs[last_shifted_i], ys[last_shifted_i], zs[last_shifted_i]))

                if index_setA == 1:
                    point_tag_list_3.append(fluxMesh.add_point(xs[shifted_i], ys[shifted_i], zs[shifted_i]))
                    point_tag_list.append(fluxMesh.add_point(xs[next_shifted_i], ys[next_shifted_i], zs[next_shifted_i]))
                    point_tag_list_2.append(fluxMesh.add_point(xs[last_shifted_i], ys[last_shifted_i], zs[last_shifted_i]))
                    simIO.log.info("ERROR: 2nd set closest to axis, this shouldn't happen?")

                if index_setA == 2:
                    point_tag_list_2.append(fluxMesh.add_point(xs[shifted_i], ys[shifted_i], zs[shifted_i]))
                    point_tag_list_3.append(fluxMesh.add_point(xs[next_shifted_i], ys[next_shifted_i], zs[next_shifted_i]))
                    point_tag_list.append(fluxMesh.add_point(xs[last_shifted_i], ys[last_shifted_i], zs[last_shifted_i]))
        else:
            # no shifting necessary for single subset
            for i in range(NTHETA):
                point_tag_list.append(fluxMesh.add_point(xs[i], ys[i], zs[i]))
        ####################################


        # Close each loop by copying 1st point to end
        point_tag_list.append(point_tag_list[0])
        # Create a closed loop from a spline of the points, add to subset loop list
        spline_tag = fluxMesh.add_bspline(point_tag_list)
        curveLoop_tag = fluxMesh.add_curve_loop([spline_tag])
        curveLoop_tag_list.append(curveLoop_tag)

        if NUM_SUBSETS == 3:
            # Close, create spline & append to local lists for each subset
            point_tag_list_2.append(point_tag_list_2[0])
            spline_tag_2 = fluxMesh.add_bspline(point_tag_list_2)
            curveLoop_tag_2 = fluxMesh.add_curve_loop([spline_tag_2])
            curveLoop_tag_list_2.append(curveLoop_tag_2)

            point_tag_list_3.append(point_tag_list_3[0])
            spline_tag_3 = fluxMesh.add_bspline(point_tag_list_3)
            curveLoop_tag_3 = fluxMesh.add_curve_loop([spline_tag_3])
            curveLoop_tag_list_3.append(curveLoop_tag_3)

    simIO.log.info('FINISHED CALCULATING CURVE LOOPS!')


    # CONCATENATE THE SUBSETS
    if NUM_SUBSETS == 3:
        combined_curveLoop_tag_list = curveLoop_tag_list_3 + curveLoop_tag_list_2 + curveLoop_tag_list
    else: # just a single (sub)set
        combined_curveLoop_tag_list = curveLoop_tag_list

    # Close the curve loop with the first curve
    # combined_curveLoop_tag_list.append(combined_curveLoop_tag_list[0])

    #SUBSET_NUM = 3 #######
    assert SUBSET_TO_MESH <= NUM_SUBSETS, 'SUBSET_TO_MESH is greater than NUM_SUBSETS'
    if NUM_SUBSETS > 1:
        if SUBSET_TO_MESH == 0:
            start = 0
            end = NUM_SUBSETS*NPHI + 1
            gmsh_filename += '_ALLof3'
        else:
            start = (SUBSET_TO_MESH-1)*NPHI
            end = SUBSET_TO_MESH*NPHI + 1
            gmsh_filename += '_{:d}of{:d}'.format(SUBSET_TO_MESH, NUM_SUBSETS) # '_'+str(SUBSET_TO_MESH)+'of3'

    elif NUM_SUBSETS == 1:
        start = 0
        end = NPHI + 1

    if FILE_TAG: gmsh_filename += '_' + FILE_TAG

    #sections = fluxMesh.add_thru_sections(combined_curveLoop_tag_list[start:end], smoothing=True)
    new_start = start+OFFSET
    new_end = end+OFFSET
    new_indices = np.arange(new_start, new_end) % len(combined_curveLoop_tag_list)

    simIO.log.info(f'new_indices: {new_indices}')
    new_curveLoop_tag_list = [combined_curveLoop_tag_list[i] for i in new_indices]

    sections = fluxMesh.add_thru_sections(new_curveLoop_tag_list, smoothing=True)

    #surface1 = fluxMesh.add_thru_sections(curveLoop_tag_list, smoothing=False)
    #surface1 = fluxMesh.add_thru_sections(curveLoop_tag_list_2, smoothing=True)
    #surface1 = fluxMesh.add_thru_sections(curveLoop_tag_list_3, smoothing=True)


    simIO.log.info('FINISHED ADDING THRU SECTIONS!')

    # Synchronize and generate 2d mesh (RUN DAT GMSH!)
    fluxMesh.synchronize()
    gmsh.model.mesh.generate(2)

    simIO.log.info('FINISHED GENERATING MESH')

    # SAVE OUTPUT!
    directory = 'output/'+ANLYS_DIR+'/data/'+ANLYS_SUBDIR+'/'
    gmsh.write(f'{directory}/{gmsh_filename}.step')


def output_debug_fig(phi, thetas, rads, all_shifted_ind, closest_to_zero, simIO):
    """ To maintain continuity of the sets, we  find the subset closest to the axis
    he islands are moving CW with +phi, so the 1st island in the list becomes the
    ast in the list once it wraps around the axis. However, its still the closest distance
    o the axis, so we can use this to determine the order of the subsets"""
    chunk_colors = ['red', 'green', 'blue']
    if closest_to_zero == 0:
        chunk_colors = ['red', 'green', 'blue']
    elif closest_to_zero == 1:
        chunk_colors = ['blue', 'red', 'green']
        simIO.log.info('ERROR!: AVG_DIST_2 IS CLOSEST TO ZERO?')
    elif closest_to_zero == 2:
        chunk_colors = ['green', 'blue', 'red']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_title('Phi angle:{:.2f} degrees'.format(phi), loc='left', fontsize=8)
    ax.set_rlim(0, 0.19)
    ax.set_rticks([0.04, 0.08, 0.12, 0.16])
    # plot lines, 
    for these_shifted_ind, thisColor in zip(all_shifted_ind, chunk_colors):
        # plot lines, start pts, and end pts, the the start position and directionality
        ax.plot(thetas[these_shifted_ind], rads[these_shifted_ind], linestyle='-',  marker='x', color=thisColor, markersize=0.3, linewidth=0.5)
        ax.scatter(thetas[these_shifted_ind[0]], rads[these_shifted_ind[0]], color='k', s=2.0)
        ax.scatter(thetas[these_shifted_ind[-1]], rads[these_shifted_ind[-1]], color='orange', s=2.0)
    simIO.saveFig(ANLYS_SUBDIR+'/debugPlot_{}.png'.format(phi), dpi=300)
    plt.close()
    return "Debug figure plotted and saved successfully."


def set_closest_to_zero(thetas, all_shifted_ind, NUM_SUBSETS):
    """ To maintain continuity of the sets, we find the subset closest to the axis
    and set that as the 'first' subset.
    The islands are moving CW with +phi, so the 1st island in the list becomes the
    last in the list once it wraps around the axis. However, its still the closest distance
    to the axis, so we can use this to determine the order of the subsets"""
    avg_dist_arr = np.empty([NUM_SUBSETS])

    # Calculate the average distance of each subset from zero
    dist_from_zero = np.minimum(thetas, 2*np.pi - thetas)

    # avg_dist_arr[0] = np.mean(dist_from_zero[:NTHETA])
    # avg_dist_arr[1] = np.mean(dist_from_zero[NTHETA:2*NTHETA])
    # avg_dist_arr[2] = np.mean(dist_from_zero[2*NTHETA:3*NTHETA])

    for subset_index in range(NUM_SUBSETS):
       avg_dist_arr[subset_index]  = np.mean(dist_from_zero[all_shifted_ind[subset_index]])
    
    # find the index of the minimum value in avg_dist_list
    return np.argmin(avg_dist_arr).astype(int)


if __name__ == '__main__':

    ## RUN DIRECTORY AND SUBDIRECTORY
    #ANLYS_DIR       = "Mar14FIT_89at360_2000sing_1p49e12_2p49e9"
    #ANLYS_DIR       = "Iota3_1500spins_atole-8_reversedHelicalCurrent"

    #ANLYS_SUBDIR    = 'ALIGNED_3x60x360mesh_toMagAxis_minus180'
    #ANLYS_SUBDIR    = "ALIGNED_3x60x180mesh_toMagAxis"

    ANLYS_DIR = "AcceptedIota3_1500spins_atole-9"
    ANLYS_SUBDIR = 'LCFS18_3x60x60mesh_s5e-6'

    #ANLYS_DIR = "ChangeToIota3_1500spins_atole-9"
    #ANLYS_SUBDIR = 'LCFS18_3x60x60mesh_s5e-6'



    ## FILE PROPERTIES (should be reading these from file)
    MAX_NUM_SUBSETS = 3
    NPHI_MESH = 60
    NTHETA_MESH = 60

    # DESIRED MESH RESOLUTION
    NPHI   = 60 #72
    NTHETA = 30
    # LCFS:18,  ISLANDS:~40-58, ISLAND AXIS: ~47, #of SURFACES: 89(?)
    # NOT WORKING EITHER WAY: 18, 38, 39, 60, 59
    SURFACE_INDX    = 19    # desired surface # to mesh
    NUM_SUBSETS     = 1     # number of islands in the surface
    SUBSET_TO_MESH  = 0     # subset# to mesh (1 to NUM_SUBSETS), 0 for all subsets
    OFFSET          = 0 #10#5     # # of indices to offset the meshing

    FILE_TAG        = "scaledx100" #'NoSmooth_Newoffset_'+str(OFFSET)  # string, additional tag for the gmsh output file name
    DEBUG_PLOT      = True           # plot debug figures


    main()