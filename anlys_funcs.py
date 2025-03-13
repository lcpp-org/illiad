
import logging
from time import perf_counter
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
from math import degrees
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

from functools import partial
import concurrent.futures as cf
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


from utility.coordtrans import XYZ_to_RTP, RTP_to_XYZ, axisShift

import phi_events


def identifyLCFS(LCFStype='inner', iconds=[0], t_maxs=[100], outputHandler=logging.getLogger(), num=11):
    """Function returns the index of the Last-Closed Flux Surface, with the option
        to input it directly, or determine it as the outermost confined surface, 
        or the confined surface innward from the first unconfined surface. """
    
    LCFStypes = ['inner', 'outer', 'input']
    if LCFStype not in LCFStypes:
        raise ValueError("Invalid LCFS type. Expected one of: %s" % LCFStypes)

    elif LCFStype == 'input':
            ## Manually select LCFS index
            LCFS_index = num 

    elif LCFStype == 'inner':
        # Assuming surfaces are ordered from 'out' to 'in':
        ## This returns the LCFS 'inside' ALL open flux surfaces
        maxTime = np.max(t_maxs)
        openSurface_ind = [i for i, t in enumerate(t_maxs) if t != maxTime] # Get indices of open flux surfaces
        LCFS_index = max(openSurface_ind) + 1

        plt.figure()
        plt.plot(iconds, t_maxs, '-o', c='k')
        plt.plot(iconds[LCFS_index], maxTime, '^', c='b')

        plt.title(r'Connection length vs. $r_{initial} (@{}\phi=324\degree)$')
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.xlabel('Minor radius [m]')
        plt.xlabel('Connection length [m]')
        outputHandler.saveFig('connectLengths')
        plt.close()

    elif LCFStype == 'outer':
        ## This returns the most 'outer' LCFS
        maxTime = np.max(t_maxs)
        LCFS_index = t_maxs.index(maxTime)

        outputHandler.log.info('LCFS_index={}'.format(LCFS_index))
        
        plt.figure()
        plt.plot(iconds, t_maxs, '-o', c='k')
        plt.plot(iconds[LCFS_index], maxTime, '^', c='b')

        plt.title(r'Connection length vs. $r_{initial} (@{}\phi=324\degree)$')
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.xlabel('Minor radius [m]')
        plt.xlabel('Connection length [m]')
        outputHandler.saveFig('connectLengths')
        plt.close()

    outputHandler.log.info('LCFS_index = {}'.format(LCFS_index))

    return LCFS_index


def Output_Poincare(iter, field_, Pdata, anlys_name, outputHandler=logging.getLogger(), saveData=True):
    """Function to output Poincare Plots and data set at a given Phi angle"""
    num_sets = len(Pdata)
    rminor = field_.a
    rmajor = field_.R0
    n, phi_ = iter

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    maxLength = 0
    for i in range(num_sets):
        maxLength = max(maxLength, len(Pdata[i][n]))

    # Looping over each initial condition
    scatter_points = np.full([num_sets, 2, maxLength], fill_value=np.nan)
    for i in range(num_sets):
        t_pts = Pdata[i][n]
        point_total = max(0, len(t_pts)-1)

        r_f = np.zeros(point_total)
        th_f = np.zeros(point_total)
        ph_f = np.zeros(point_total)

        for j in range(point_total):
            r_f[j], th_f[j], ph_f[j] = XYZ_to_RTP(t_pts[j][:3], rmajor)

        if saveData:
            scatter_points[i][0][:th_f.size] = th_f
            scatter_points[i][1][:r_f.size] = r_f
        else:
            pass

        plt.scatter(th_f, r_f, marker='.', s=1.00, c='k', linewidths=0.0)

    if saveData:
        f_output = scatter_points
        fname = anlys_name + '_{:03.0f}'.format(degrees(phi_))
        outputHandler.saveNumpyData(f_output, fname)
    else:
        pass 

    ax.set_rmax(rminor)
    ax.set_rticks(np.arange(0.0, 0.19, 0.02))
    ax.yaxis.set_tick_params(labelsize=5)
    ax.grid(linewidth = 0.25, linestyle=':', c='k')
    phi_phys = (phi_ + (198 * np.pi/180.)) % (2*np.pi)  
    plt.title('$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format(phi_phys*180/np.pi, phi_*180/np.pi), loc='left')
    plot_name = anlys_name +'/'+ anlys_name + '_phi={:03.0f}.png'.format(phi_*180/np.pi)
    outputHandler.saveFig(plot_name, dpi=300)
    plt.close()

    return '\tPHI: {}'.format(phi_*(180/np.pi))


def boris_wrapper(ion_list, b_hidra, ion_temp_eV, dt, tmax, dr_String):
    ion_ = ion_list[0]
    log = logging.getLogger()
    log.info('###########################################################################')
    log.info('RUNNING BORIS-BUNEMAN SOLVER WITH NEW ION SEED POINTS:')
    log.info('Initial Conditions:\t{} points'.format(len(ion_list)))
    log.info('Ions:\tmass={}[amu], q={}[Coulomb], ion temp.={}[eV]'#, initial velocity={:.0f} [m/s]'
             .format(ion_.mass, ion_.charge, ion_temp_eV))#, init_v_phi))
    log.info('Shells generated at delta-r(s) of {}mm from LCFS'.format(dr_String))
    log.info('SOLVER SETTINGS: tmax: {}sec., dt: {}sec., N={}pts'.format(tmax, dt, int(tmax/dt)))
    log.info('###########################################################################\n')


    ## PARALLELIZATION WITH CONCURRENT FUTURES 'MAP' OVER EACH PARTICLE
    boris_x = partial(boris_solver, dt=dt, tmax=tmax, Bfield=b_hidra)
    t_start = perf_counter()
    with cf.ProcessPoolExecutor(max_workers=6) as executor:
        boris_output_ = executor.map(boris_x, ion_list)#, chunksize=2)
    t_stop = perf_counter()
    tot_elapsed_time = t_stop - t_start
    log.info('ALL SOLVERS FINISHED IN {} seconds\n###############\n\n'.format(tot_elapsed_time))

    return boris_output_



## TESTNG!!!
def boris_solver2(ions, dt, tmax, Bfield):
    """Function to take in a particle and field object and solves the particle path until termination even or tmax
       using a fixed-step Boris-Buneman Solver, based on (Birdsall, 4-3&4)"""
    log = logging.getLogger()
    log.info( 'Start ICs: {}-{}'.format(ions[0].particleID, ions[-1].particleID) )
    t_startInd = perf_counter()

    Nparticles = torch.int
    Nsteps = torch.int
    Nparticles = len(ions)
    Nsteps = int((tmax // dt) + 1)

    tvec = torch.zeros([Nparticles, 3], dtype=torch.float64).to(device)
    svec = torch.zeros([Nparticles, 3], dtype=torch.float64).to(device)
    #v_k = torch.zeros([Nparticles, 3]).to(device)
    vprime = torch.zeros([Nparticles, 3], dtype=torch.float64).to(device)
    vplus = torch.zeros([Nparticles, 3], dtype=torch.float64).to(device)
    wallPts = torch.zeros([Nparticles, 3], dtype=torch.float64).to(device)
    tmag = torch.zeros(Nparticles, dtype=torch.float64).to(device)
    #tmag2 = torch.zeros(Nparticles, dtype=torch.float64).to(device)

    r_k = torch.zeros(Nparticles, dtype=torch.float64).to(device)

    ## BIG ARRAY #!#!#!#!#!#
    #histories = torch.zeros([Nsteps, Nparticles, 3], dtype=torch.float64).to(device)

    # Need particle parms: qdt2m, v0, p0
    qdt2m = torch.asarray( [ion.charge_mass_ratio*dt/2 for ion in ions] ).to(device)

    # Initialize particle states
    v_k = np.asarray([ ion.vel0_XYZ for ion in ions ])
    v_k = torch.tensor(v_k, dtype=torch.float64).to(device)

    [ion.setPosition(0, ion.pos0_XYZ) for ion in ions]
    pos_k = np.asarray( [ion.pos0_XYZ for ion in ions])
    pos_k = torch.tensor(pos_k, dtype=torch.float64).to(device)

    ## NEED v_n-1/2 TO START
    ##################
    # Get field at every particle location
    tvec = (Bfield.interpField(pos_k) * qdt2m).T
    tmag = torch.linalg.norm(tvec, axis=-1)

    vprime = v_k + torch.linalg.cross(v_k, tvec)#, axis=1) #dim=1?

    svec = 2*tvec / ( 1 + (tmag*tmag)[:,None] )# svec given by (4-4, Eq13)

    vplus = v_k - torch.linalg.cross(vprime, svec) / 2 #, axis=1) / 2 # stepping back a 1/2 step!

    v_k = vplus #.detach().clone()#?NECESSARY?

    #calculate r (of rtp) for particles
    x2 = pos_k.T[0] * pos_k.T[0]
    y2 = pos_k.T[1] * pos_k.T[1]
    z2 = pos_k.T[2] * pos_k.T[2]
    r_k = torch.sqrt( x2 + y2 + z2 + Bfield.R0*Bfield.R0 - 2*Bfield.R0*torch.sqrt(x2 + y2) ) #calculate r (of rtp) for particles

    running = torch.arange(0, Nparticles, 1, dtype=torch.int).to(device)
    Nrunning = running.size(dim=0)
    ## STEPPING THROUGH DTs
    log.info( 'START STEPPING...')
    
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm(loggers=[log]):
        pbar = tqdm(range(Nsteps-1), ncols= 100, mininterval=1.0)
        for k in pbar:
            tvec[running] = (Bfield.interpField(pos_k[running]) * qdt2m[running]).T #tvec given by (4-4, Eq11)
            tmag[running]  = torch.linalg.norm(tvec[running], axis=-1)
            #tmag2[running] = tmag2[running][0] * tmag2[running][0] + tmag2[running][1] * tmag2[running][1] + tmag2[running][2] * tmag2[running][2]

            vprime[running]  = v_k[running]  + torch.linalg.cross(v_k[running], tvec[running])#, axis=1)# vminus is incremented (4-4, Eq10), get vprime

            svec[running]  = 2*tvec[running]  / ( 1 + (tmag[running]*tmag[running])[:,None] )# svec given by (4-4, Eq13)
            #svec[running]  = 2*tvec[running]  / ( 1 + tmag2[running][:,None] )# svec given by (4-4, Eq13)

            vplus[running]  = v_k[running]  + torch.linalg.cross(vprime[running], svec[running])#, axis=1)# from vminus, vprime, svec (4-4, Eq12), get vplus 

            v_k[running]  = vplus[running] #.detach().clone() #?NECESSARY?

            pos_k[running]  = pos_k[running]  + v_k[running] * dt

            ### HISTORY OFF!! ===> SPEED!!!  #!#!#!#!#!#
            #[ ion.setPosition(k+1, pos_k[ndex][:3]) for ndex, ion in enumerate(ions) ]
            #histories[k] = pos_k

            #calculate r (of rtp) for particles
            x2[running] = pos_k[running].T[0]# * pos_k.T[0,running]
            y2[running] = pos_k[running].T[1]# * pos_k.T[1,running]
            z2[running] = pos_k[running].T[2]# * pos_k.T[2,running]
            x2[running] = x2[running]*x2[running]
            y2[running] = y2[running]*y2[running]
            z2[running] = z2[running]*z2[running]

            r_k[running] = torch.sqrt( x2[running] + y2[running] + z2[running] + Bfield.R0*Bfield.R0 - 2*Bfield.R0*torch.sqrt(x2[running] + y2[running]) ) #calculate r (of rtp) for particles

            # remove terminated particles from running list
            running = torch.where(r_k<Bfield.a)
            Nrunning = running[0].size(dim=0)

            # update progress bar
            pbar.set_postfix({'#Particles running': Nrunning})

    # POSTPONE THIS UNTIL AFTER LOOP COMPLETION
    #find indices of particles that have intersected the vacuum vessel
    terminated = torch.where(r_k>=Bfield.a)
    # update list of wall intersection points
    wallPts[terminated] = pos_k[terminated]


    t_stopInd = perf_counter()

    elapsed_timeInd = t_stopInd - t_startInd
    min_, sec_ = divmod(elapsed_timeInd, 60)
    hr_, min_ = divmod(min_, 60)

    log.info( 'ELAPSED TIME({} Particles): {:02.0f}H:{:02.0f}M:{:02.3f}S'.format(Nparticles, hr_, min_, sec_) )

    return wallPts, pos_k

def boris_solver(ion, dt, tmax, Bfield):
    """Function to take in a particle and field object and solves the particle path until termination even or tmax
       using a fixed-step Boris-Buneman Solver, based on (Birdsall, 4-3&4)"""
    log = logging.getLogger()
    log.info('Start IC: {}, {}'.format(ion.particleID, ion.pos0_XYZ))
    t_startInd = perf_counter()

    B = np.empty(3, dtype=np.float64)
    wallPt = np.zeros(3)
    N = int((tmax // dt) + 1)
    # Need particle parms: qdt2m, v0, p0
    qdt2m = ion.charge_mass_ratio * dt/2
    #print(f'{qdt2m=}')
    ion.setPosition(0, ion.pos0_XYZ)


    v_k = ion.vel0_XYZ

    # Need v_n-1/2 to start
    B, dum_ = Bfield.interpField(ion.pos_XYZ[0])

    tvec = qdt2m * B# tvec given by (4-4, Eq11)

    vprime = v_k + np.array([v_k[1]*tvec[2] - v_k[2]*tvec[1], 
                         v_k[2]*tvec[0] - v_k[0]*tvec[2],
                         v_k[0]*tvec[1] - v_k[1]*tvec[0]])   #np.cross(v_k, tvec)# vminus is incremented (4-4, Eq10), get vprime
    
    svec = 2*tvec / ( 1 + (np.linalg.norm(tvec)*np.linalg.norm(tvec)) )# svec given by (4-4, Eq13)

    vplus = v_k -  np.array([vprime[1]*svec[2] - vprime[2]*svec[1], 
                            vprime[2]*svec[0] - vprime[0]*svec[2],
                            vprime[0]*svec[1] - vprime[1]*svec[0]]) / 2 # from vminus, vprime, svec (4-4, Eq12), get vplus 

    v_k = vplus

    ## STEPPING THROUGH DTs
    for k in range(N-1):
        B, dum_ = Bfield.interpField(ion.pos_XYZ[k])

        tvec = qdt2m * B# tvec given by (4-4, Eq11)

        #vprime = v_k + np.cross(v_k, tvec)# vminus is incremented (4-4, Eq10), get vprime
        vprime = v_k + np.array([v_k[1]*tvec[2] - v_k[2]*tvec[1], 
                                 v_k[2]*tvec[0] - v_k[0]*tvec[2],
                                 v_k[0]*tvec[1] - v_k[1]*tvec[0]])   #np.cross(v_k, tvec)# vminus is incremented (4-4, Eq10), get vprime

        svec = 2*tvec / ( 1 + (np.linalg.norm(tvec)*np.linalg.norm(tvec)) )# svec given by (4-4, Eq13)

        #vplus = v_k + np.cross(vprime, svec)# from vminus, vprime, svec (4-4, Eq12), get vplus 
        vplus = v_k + np.array([vprime[1]*svec[2] - vprime[2]*svec[1], 
                                vprime[2]*svec[0] - vprime[0]*svec[2],
                                vprime[0]*svec[1] - vprime[1]*svec[0]]) # from vminus, vprime, svec (4-4, Eq12), get vplus 

        xplus = ion.pos_XYZ[k] + vplus*dt # from vplus, dt, get xplus
        v_k = vplus
        ion.setPosition(k+1, xplus)
        
        ion.maxLife = (k+1)*dt
        if phi_events.inVV(1, ion.pos_XYZ[k+1], Bfield) < 0.0:
            ion.terminated = True
            wallPt = ion.pos_XYZ[k+1]
            break

    t_stopInd = perf_counter()
    elapsed_timeInd = t_stopInd - t_startInd

    if ion.terminated:
        log.info('Success!: Particle {} of {} took {:.5f} sec.\tWall Event at t={:.5f}, k={}'
                 .format(ion.particleID, ion.particleCount, elapsed_timeInd, ion.maxLife, ion.maxLife//dt))
    else:
        log.info('Success!: Particle {} of {} took {:.5f} sec.\tWall Event at t='
                 .format(ion.particleID, ion.particleCount, elapsed_timeInd))
        
    return (wallPt, ion.pos_XYZ)



def find_Axis(theta_vals, r_vals, field):
    """Function to find the geometric center of a set of points in r, theta coordinates"""
    theta_size = theta_vals.size
    ## CONVERT TO 2D XZ COORDINATES
    x_in = np.empty(theta_size)
    y_in = np.empty(theta_size)
    z_in = np.empty(theta_size)
    for i, theta, in enumerate(theta_vals):
        x_in[i], y_in[i], z_in[i] = RTP_to_XYZ(np.array([r_vals[i], theta, 0.]), field.R0)

    ## FIND THE AXIS BY AVERAGING THE POSITIONS
    x_avg = np.average(x_in)
    y_avg = 0.0
    z_avg = np.average(z_in)

    axis_xyz = np.array([x_avg, y_avg, z_avg])
    axis_rtp = XYZ_to_RTP(axis_xyz, field.R0)

    return axis_rtp



def find_subsets(theta_r_pts, mag_axis, field, BINS=30):
    """Function to find contiguous subsets of points in theta-r space"""
    test_flag = False
    # make a histogram of the point density vs theta
    hist, bin_edges = np.histogram(theta_r_pts.T[0], bins=BINS, range=(0., 2*np.pi))
    dtheta_bin = bin_edges[1] - bin_edges[0]

    # find how many contiguous sets of adjacents bins there are
    non_empty_bins = np.where(hist > 0)[0]
    contiguous_sets = np.split(non_empty_bins, np.where(np.diff(non_empty_bins) != 1)[0]+1)
    # if the first and last bins are non-empty, then the first and last sets of bins are contiguous
    if hist[0] > 0 and hist[-1] > 0 and len(contiguous_sets) > 1:
        contiguous_sets[0] = np.concatenate((contiguous_sets[-1], contiguous_sets[0]))
        contiguous_sets.pop()
        test_flag = True
    num_sets = len(contiguous_sets)

    subsetData = []
    subsetCenters = np.zeros([num_sets, 2])
    # loop throught each contiguous subset
    for i, contiguous_set in enumerate(contiguous_sets):
        thisSet_tr = []
        # calculate bin bounds
        lowerBound = contiguous_set*dtheta_bin
        upperBound = lowerBound + dtheta_bin
        # append data within each bin belonging to the subset
        for lo, hi in zip(lowerBound, upperBound):
            thisSet_tr += [point for point in theta_r_pts if lo <= point[0] < hi]
        thisSet_tr = np.array(thisSet_tr)

        # sort the subset by theta
        thisSet_tr = thisSet_tr[np.argsort(thisSet_tr[:, 0])]

        # TESTING, ONLY CONSIDER 3 SUBSETS!
        # only split if there are between 3 and 5 subsets, treat rest as 1 set
        if num_sets > 2 and num_sets < 4:
            subsetCenters[i][:] = find_Axis(thisSet_tr.T[0], thisSet_tr.T[1], field)[:2]
            # shift the data to be relative to the center of the subset
            thisSetLocAxis = np.array([axisShift(r, theta, *subsetCenters[i][:2]) for theta, r in thisSet_tr])
            thisSetLocAxis = thisSetLocAxis[np.argsort(thisSetLocAxis[:, 0])]
            subsetData += [thisSetLocAxis]
        # if there is only 1 subset, or lots(noisy data), then keep the original magnetic axis
        else:
            subsetCenters[i][:] = mag_axis[:2]
            thisSetLocAxis = thisSet_tr
            subsetData = [theta_r_pts]

    return subsetData, subsetCenters, hist, bin_edges, test_flag



# append and spline data
from scipy.interpolate import make_smoothing_spline, spalde, splev, splrep
def spline_Data(theta_pts, rad_pts):
    # Copy data to both ends for pseudo-periodicity (smooth spline endpoints)
    th_size = len(theta_pts)
    append_length = int(th_size/2)
    th_A = theta_pts[append_length:-1] - 2*np.pi
    th_B = theta_pts[1:append_length] + 2*np.pi
    theta_spl = np.concatenate((th_A, theta_pts, th_B))
    rad_A = rad_pts[append_length:-1]
    rad_B = rad_pts[1:append_length]
    rad_spl = np.concatenate((rad_A, rad_pts, rad_B))
    # spline parameters
    fSurface_splineParms, res, fail, msg = splrep(theta_spl, rad_spl, k=3, s=1e-5, per=False, full_output=1, quiet=1)

    return fSurface_splineParms, res, fail, msg