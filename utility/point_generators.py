import numpy as np
import logging
from scipy.interpolate import splev, splrep, interp1d
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10})
#plt.rcParams.update({'figure.autolayout':True})

from classes.particle import Ion
from utility.coordtrans import RTP_to_XYZ, XYZ_to_RTP, RTP_XYZ_JAC, axisShift, align_z_to_vector

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _equal_area_theta_samples(surface_spline, magnetic_to_geometric_shift, ntheta,
                              major_radius, dense_count=4096):
    """Return approximately equal-area poloidal samples for one toroidal plane."""
    theta_dense = np.linspace(0.0, 2.0*np.pi, dense_count + 1)
    radius_dense = splev(theta_dense, surface_spline)
    theta_geo, radius_geo = axisShift(
        theta_dense, radius_dense, *magnetic_to_geometric_shift
    )

    major_r = major_radius + radius_geo*np.cos(theta_geo)
    vertical = radius_geo*np.sin(theta_geo)
    segment_length = np.hypot(np.diff(major_r), np.diff(vertical))
    segment_area = 0.5*(major_r[:-1] + major_r[1:])*segment_length
    cumulative_area = np.concatenate(([0.0], np.cumsum(segment_area)))

    if not np.all(np.isfinite(cumulative_area)) or cumulative_area[-1] <= 0.0:
        raise ValueError('Could not construct a finite, positive LCFS area distribution')

    cumulative_area /= cumulative_area[-1]
    targets = (np.arange(ntheta, dtype=np.float64) + 0.5) / ntheta
    return np.interp(targets, cumulative_area, theta_dense)


def _plot_emitter_spacing_comparison(surface_spline, magnetic_to_geometric_shift,
                                     theta_equal, theta_area, phi_deg, outputHandler):
    """Save one cross-section comparison of equal-theta and equal-area emitters."""
    theta_curve = np.linspace(0.0, 2.0*np.pi, 2000)
    radius_curve = splev(theta_curve, surface_spline)
    curve_geo = axisShift(theta_curve, radius_curve, *magnetic_to_geometric_shift)

    radius_equal = splev(theta_equal, surface_spline)
    equal_geo = axisShift(theta_equal, radius_equal, *magnetic_to_geometric_shift)
    radius_area = splev(theta_area, surface_spline)
    area_geo = axisShift(theta_area, radius_area, *magnetic_to_geometric_shift)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(curve_geo[1]*np.cos(curve_geo[0]), curve_geo[1]*np.sin(curve_geo[0]),
            color='black', linewidth=1.0, label='Fitted LCFS')
    ax.scatter(equal_geo[1]*np.cos(equal_geo[0]), equal_geo[1]*np.sin(equal_geo[0]),
               marker='x', s=24, color='tab:blue', label=r'Equal $\theta$')
    ax.scatter(area_geo[1]*np.cos(area_geo[0]), area_geo[1]*np.sin(area_geo[0]),
               marker='o', s=14, facecolors='none', edgecolors='tab:orange',
               label='Approx. equal area')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$R-R_0$ (m)')
    ax.set_ylabel('$Z$ (m)')
    ax.set_title(r'Emitter-spacing comparison at $\phi={:.0f}^\circ$'.format(phi_deg))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_name = 'EmitterSpacingComparison_phi={:03.0f}.png'.format(phi_deg)
    outputHandler.saveFig(plot_name, dpi=300)
    outputHandler.log.info('OUTPUT PLOT: {}'.format(plot_name))
    plt.close(fig)


def generateSeedShells(drList, Ntheta, phi_array, lcfs_index, filename, Bfield,
                       genNormals=False, Efield=None, outputHandler='simIO',
                       emitter_spacing='equal_theta'):
    """
    Generates seed points on the last closed flux surface (LCFS) for a given magnetic field configuration.

    Args:
        drList (list or np.ndarray): List of radial displacements from the LCFS for generating seed shells.
        Ntheta (int): Number of theta points to sample along the LCFS.
        phi_array (array-like): Array of toroidal angles (in degrees) at which to generate seed points.
        lcfs_index (int): Index of the LCFS in the loaded Poincare data.
        filename (str): Directory name for saving output data and plots.
        Bfield (object): Magnetic field object containing geometry and parameters (e.g., R0, a).
        genNormals (bool, optional): If True, use the local electric-field direction
            with the outward geometric LCFS normal as a fallback. Defaults to False.
        Efield (object, optional): Electric field used to define the preferred launch
            direction when ``genNormals`` is True.
        outputHandler (object, optional): Handler for logging, saving data, and figures. Defaults to 'simIO'.
        emitter_spacing (str, optional): ``equal_theta`` for the legacy angular grid or
            ``equal_area`` for approximate equal-area poloidal sampling. Defaults to
            ``equal_theta``.

    Returns:
        tuple: A tuple containing:
            - seed_list (list): List of generated seed point coordinates in Cartesian (XYZ) space.
            - normals_list (list): List of normal vectors at each seed point (if genNormals is True), otherwise empty.
    """
    seed_list = []
    normals_list = []
    emitter_spacing = str(emitter_spacing).lower()
    if emitter_spacing not in {'equal_theta', 'equal_area'}:
        raise ValueError("emitter_spacing must be 'equal_theta' or 'equal_area'")
    outputHandler.createSubDir(filename)
    outputHandler.log.info('GENERATING SEED POINTS FOR LCFS INDEX: {}'.format(lcfs_index))
    outputHandler.log.info('EMITTER SPACING: {}'.format(emitter_spacing))

    for phi_gen_deg in phi_array:
        input_filename = 'Poincare_{:03.0f}.npy'.format(phi_gen_deg)
        th_in, r_in = outputHandler.loadNumpyData(input_filename, mmap_mode='r')[lcfs_index]
        r_in = r_in[~np.isnan(r_in)]
        th_in = th_in[~np.isnan(th_in)]

        phi_deg = int(phi_gen_deg)
        phi_rad = phi_gen_deg * np.pi / 180.

        # hack solution, need to determine why an extra 30 copies of 1 initial condition are being appended to this event
        if phi_deg == 324:
            r_in = r_in[30:]
            th_in = th_in[30:]
        th_size = th_in.size

        # find the centroid(?) by average positions
        x_in = np.empty(th_size)
        z_in = np.empty(th_size)
        for i, theta, in enumerate(th_in):
            x_in[i], y_in, z_in[i] = RTP_to_XYZ(np.array([r_in[i], theta, 0.]), Bfield.R0)

        x_avg = (np.max(x_in) + np.min(x_in))/2
        z_avg = (np.max(z_in) + np.min(z_in))/2
        XYZ_delta = np.array([x_avg, 0.0, z_avg])

        # shift origin of r, theta coords from geo center to magnetic axis, sort pts on theta
        RTP_delta = XYZ_to_RTP(XYZ_delta, Bfield.R0)[1::-1]
        RTP_delta_rev = np.copy(RTP_delta)
        RTP_delta_rev[0] += np.pi
        magCenterCoords = np.empty((th_size, 2))
        for i, theta, in enumerate(th_in):
            magCenterCoords[i] = axisShift(theta, r_in[i], *RTP_delta)
        sortedMagCenter = magCenterCoords[np.argsort(magCenterCoords[:,0])]

        # Append data to either end for pseudo-periodicity (smooth spline endpoints)
        theta_pts = sortedMagCenter.T[0]
        rad_pts = sortedMagCenter.T[1]
        append_length = int(th_size/2)
        th_A = np.copy(theta_pts[append_length:-1]) - 2*np.pi
        rad_A = np.copy(rad_pts[append_length:-1])
        th_B = np.copy(theta_pts[1:append_length]) + 2*np.pi
        rad_B = np.copy(rad_pts[1:append_length])

        theta_spl = np.concatenate((th_A, theta_pts, th_B))
        rad_spl = np.concatenate((rad_A, rad_pts, rad_B))

        ## SPLINING
        #fSurface_splineParms = splrep(theta_spl, rad_spl, s=1e-4, k=3, per=False, quiet=1)
        fSurface_splineParms = splrep(theta_spl, rad_spl, s=8e-6, k=3, per=False, quiet=1) # 's' from fluxTest4.py
        theta_equal = np.linspace(0, 2*np.pi*(1 - 1/Ntheta), Ntheta)
        if emitter_spacing == 'equal_area':
            theta_evals = _equal_area_theta_samples(
                fSurface_splineParms, RTP_delta_rev, Ntheta, Bfield.R0
            )
            if phi_gen_deg == phi_array[0]:
                _plot_emitter_spacing_comparison(
                    fSurface_splineParms, RTP_delta_rev, theta_equal, theta_evals,
                    phi_gen_deg, outputHandler,
                )
        else:
            theta_evals = theta_equal
        seedPts_0 = splev(theta_evals, fSurface_splineParms)
        derivs =  splev(theta_evals, fSurface_splineParms, der=1)

        # Geometry-based surface normals used when the electric-field direction is invalid.
        # LCFS in poloidal plane is parameterized by r(θ). In the cross-section (x,z):
        #   x(θ) = r cosθ, z(θ) = r sinθ
        # tangent t = d/dθ[x,z], normal n = [dz/dθ, -dx/dθ]. Choose outward via dot(n, e_r)>0.
        r0 = np.asarray(seedPts_0, dtype=np.float64)
        drdth = np.asarray(derivs, dtype=np.float64)
        th = np.asarray(theta_evals, dtype=np.float64)
        dx_dth = drdth * np.cos(th) - r0 * np.sin(th)
        dz_dth = drdth * np.sin(th) + r0 * np.cos(th)
        nx = dz_dth
        nz = -dx_dth
        # Enforce outward direction
        erx = np.cos(th)
        erz = np.sin(th)
        outward = (nx * erx + nz * erz) >= 0
        nx = np.where(outward, nx, -nx)
        nz = np.where(outward, nz, -nz)
        n2_norm = np.sqrt(nx * nx + nz * nz)
        n2_norm = np.where(n2_norm > 0, n2_norm, 1.0)
        nxu = nx / n2_norm
        nzu = nz / n2_norm
        # Rotate cross-section normal into XYZ at this phi (note sign convention matches RTP_to_XYZ)
        geom_normals = np.stack([
            nxu * np.cos(phi_rad),
            (-1.0) * nxu * np.sin(phi_rad),
            nzu,
        ], axis=1)

        ## START FIGURE FOR PLOTTING
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

        ## PLOTTING THE SPLINE FIT
        thetaPlot = np.linspace(0., 2*np.pi, 5000)
        rPlot = splev(thetaPlot, fSurface_splineParms)
        geoCenterCoords = axisShift(thetaPlot, rPlot, *RTP_delta_rev)
        ax.plot(*geoCenterCoords, '-k', linewidth=0.8) # fitted spline curve

        output_ind     = np.zeros((Ntheta, 3))
        output_ind_geo = np.zeros((Ntheta, 3))
        output_ind_XYZ = np.zeros((Ntheta, 3))
        output_ind_normal = np.zeros((Ntheta, 3))
        plot_norm_rtp = np.zeros((Ntheta, 3))
        plot_norm_rtp_REF = np.zeros((Ntheta, 3))
        outData = []
        outNormals = []
        n_invalid_normals = 0
        n_geom_fallback = 0
        n_default_fallback = 0
        tensor_ind_XYZ = torch.zeros((Ntheta, 3), dtype=torch.float32, device=device)
        for dr in drList:
            for i, theta in enumerate(theta_evals):
                # scale delta-r to achieve uniform expansion normal to surface
                adj_dr = dr * np.sqrt(1 + derivs[i]**2)
                seedPt = seedPts_0[i] + adj_dr

                # shift back to geometric axis
                output_ind_geo[i][:2] = axisShift(theta, seedPt, *RTP_delta_rev)
                output_ind_geo[i][1] = min(Bfield.a, output_ind_geo[i][1])
                output_ind_geo[i][2] = phi_rad # keep phi constant for all points in this shell

                # convert rtp vector to xyz
                output_ind[i] = np.array([output_ind_geo[i][1], output_ind_geo[i][0], output_ind_geo[i][2]])
                output_ind_XYZ[i] = RTP_to_XYZ(output_ind[i], Bfield.R0)
                tensor_ind_XYZ[i] = torch.tensor(output_ind_XYZ[i], dtype=torch.float32, device=device)

                if genNormals:
                    vec = Efield.interpField(tensor_ind_XYZ[i], Cart=True).cpu().numpy()
                    vec_norm = np.linalg.norm(vec)
                    if np.isfinite(vec_norm) and vec_norm > 0:
                        output_ind_normal[i] = vec / vec_norm
                    else:
                        geom_normal = geom_normals[i]
                        geom_norm = np.linalg.norm(geom_normal)
                        if np.isfinite(geom_norm) and geom_norm > 0:
                            output_ind_normal[i] = geom_normal / geom_norm
                            n_geom_fallback += 1
                        else:
                            output_ind_normal[i] = np.array([0.0, 0.0, 1.0])
                            n_default_fallback += 1
                        n_invalid_normals += 1

            ## PLOTTING THE SEED POINTS
            ax.plot(*output_ind_geo.T[:2], '--ok', linewidth=0.25, markersize=2)
            outData.extend(np.copy(output_ind_XYZ))
            if genNormals:
                outNormals.extend(np.copy(output_ind_normal))

        if genNormals:
            if n_invalid_normals > 0:
                outputHandler.log.warning(
                    f"{filename}: {n_invalid_normals}/{Ntheta*len(drList)} electric-field launch directions were invalid at phi={phi_deg} deg; "
                    f"using geometric fallback for {n_geom_fallback} and default [0,0,1] for {n_default_fallback}"
                )
            # CONVERT NORMAL VECTORS TO RTP FOR PLOTTING
            for i in range(len(output_ind_normal)):
                plot_norm_rtp[i] = RTP_XYZ_JAC(output_ind_geo[i], output_ind_normal[i], form='xyz2rtp')



            # Plot semicircles with normal as pole
            semicircle_radius = 0.009  # Adjust size as needed
            n_circle_points = 12       # Number of points for smooth semicircle
            
            for i in range(len(output_ind_geo)):
                center_theta, center_r = output_ind_geo[i][:2]
                normal_theta_component, normal_r_component = plot_norm_rtp[i][:2]

                # Calculate normal direction angle in polar coordinates
                normal_direction = np.arctan2(normal_r_component, normal_theta_component)
                
                # Create semicircle oriented ±90° to the normal
                perp_angle = normal_direction #+ np.pi/2
                start_angle = perp_angle - np.pi/2
                end_angle = perp_angle + np.pi/2
                angle_range = np.linspace(start_angle, end_angle, n_circle_points)

                # Calculate semicircle points in Cartesian coordinates, then convert back
                x_circle = center_r * np.cos(center_theta) + semicircle_radius * np.cos(angle_range)
                y_circle = center_r * np.sin(center_theta) + semicircle_radius * np.sin(angle_range)
                
                # Convert back to polar
                r_circle = np.sqrt(x_circle**2 + y_circle**2)
                theta_circle = np.arctan2(y_circle, x_circle)
                theta_circle = theta_circle % (2 * np.pi) # Ensure positive theta values

                lcfsfit = interp1d(geoCenterCoords[0], geoCenterCoords[1], fill_value="extrapolate")
                r_lower = lcfsfit(theta_circle)        

                # Filter out points outside the domain
                valid_points = (r_circle <= Bfield.a) & (r_circle >= 0)

                if np.any(valid_points):
                    ax.plot(theta_circle[valid_points], r_circle[valid_points], ':',
                            color='blue', linewidth=0.8, alpha=0.8)
                    ax.fill_between(theta_circle[valid_points], r_circle[valid_points], r_lower[valid_points], color='lightblue', alpha=0.8)

            ## PLOT NORMAL VECTORS
            ax.quiver(*output_ind_geo.T[:2], *plot_norm_rtp.T[:2],  color='blue', scale=18, width=0.004, angles='uv')


        ax.set_rmax(Bfield.a)
        ax.grid(linewidth = 0.25, linestyle=':', c='k')
        ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                            labels=['Low\nField', '', '', '', 'High\nField', '', '', ''], fontsize=14)
        ax.tick_params(pad=10)
        ax.set_rgrids([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
                        labels=[], angle=0, fontsize=4)
        #plt.title(r'$\phi$={:02.0f}$\degree$'.format(phi_deg), loc='left')
        plt.tight_layout()
        plot_name = filename+'/'+'InitConds_phi={:03.0f}.png'.format(phi_deg)
        outputHandler.saveFig(plot_name)
        plt.close()

        outArray = np.asarray(outData)
        outputHandler.saveNumpyData(outArray, filename)

        if genNormals: outNormalsArray = np.asarray(outNormals)
    
        seed_list.extend(outArray)
        normals_list.extend(outNormalsArray)

    outputHandler.log.info('FINISHED LOADING NUMPY DATA & GENERATING INIT. POSITIONS\n')
    return seed_list, normals_list


def generate_MB_velocities(N_particles, normals_list, ion_temp, ion_mass, nparticles_per_emitter=1, outputHandler='simIO'):
    """
    Generates initial velocities for particles following a Maxwell-Boltzmann energy distribution.

    The function samples a cosine-weighted angular distribution over a hemisphere and scales the
    velocities according to the specified ion temperature. The resulting velocity vectors are
    stored in emitter-major order and rotated to align with the provided surface normals.

    Args:
        N_particles (int): Number of particles to generate velocities for.
        normals_list (array-like): List or array of normal vectors to align velocities with.
        ion_temp (float): Ion temperature in eV.
        ion_mass (float): Ion mass in atomic mass units (amu).
        nparticles_per_emitter (int, optional): Number of particles per emitter. Defaults to 1.

    Returns:
        np.ndarray: Array of shape (N_particles, 3) containing the generated velocity vectors.
    """
    outputHandler.log.info(f'GENERATING INITIAL VELOCITIES (MAXWELLIAN DIST., T={ion_temp}eV):')
    kg_per_amu = 1.660_539_068E-27
    kboltz = 1.602_176_634E-19 # Joules/eV
    # Uniform sampling of the projected unit disk gives the cosine-weighted
    # hemispherical distribution p(alpha) = 2 sin(alpha) cos(alpha).
    r = np.sqrt(np.random.uniform(0, 1, N_particles))

    z = np.sqrt(1 - r**2)
    phi = np.random.uniform(0, 2*np.pi, N_particles)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    velocity_array = np.stack([x, y, z], axis=1) # shape (N, 3)

    # GENERATE NORMAL DISTRIBUTION OF SPEEDS
    # Calculate the root mean square velocities
    v_rms1d = np.sqrt( kboltz*ion_temp / (ion_mass*kg_per_amu) )
    initSpeeds = v_rms1d * np.sqrt(np.random.chisquare(df=3, size=N_particles))

    # APPLY INIT SPEEDS TO THE RANDOM UNIT VECTORS
    velocity_array *= initSpeeds[:, None]

    # ROTATE TO ALIGN POLE WITH NORMAL VECTOR
    n_bad_normals = 0
    for i, normal in enumerate(normals_list):
        normal = np.asarray(normal, dtype=np.float64)
        if normal.shape != (3,) or (not np.all(np.isfinite(normal))) or np.linalg.norm(normal) < 1e-15:
            Rotater = np.eye(3)
            n_bad_normals += 1
        else:
            Rotater = align_z_to_vector(normal)
        start_idx = i * nparticles_per_emitter
        end_idx = start_idx + nparticles_per_emitter
        velocity_array[start_idx:end_idx] = velocity_array[start_idx:end_idx] @ Rotater.T

    if n_bad_normals > 0:
        outputHandler.log.warning(
            f"generate_MB_velocities: {n_bad_normals}/{len(normals_list)} normals were zero/invalid; "
            "skipped rotation (used identity) for those emitters"
        )

    return velocity_array


def ionInitializer(initial_conditions, ion_properties, bfield, efield, outputHandler='simIO',
                   return_normals=False, emitter_spacing='equal_theta',
                   save_emitter_locations=False):

    mass, charge, temperature = ion_properties
    lcfs_index, nphi, ntheta, deltrs, nparticles_per_emitter = initial_conditions

    n_emitters = len(deltrs) * ntheta * nphi
    n_particles = n_emitters * nparticles_per_emitter
    delimiter = '-'
    dr_String = delimiter.join(str(int(dr*1000)) for dr in deltrs)

    ## GENERATE INITIAL POSITIONS
    phiGen_arr = np.linspace(360.0 / nphi, 360.0, nphi).tolist()
    seed_list, normals_list =  generateSeedShells(deltrs, ntheta, phiGen_arr, lcfs_index, 'IonSeedPts_{}mm'.format(dr_String),
                                                    bfield, Efield=efield, genNormals=True, outputHandler=outputHandler,
                                                    emitter_spacing=emitter_spacing)
    if save_emitter_locations:
        emitter_locations = np.asarray(seed_list, dtype=np.float64).reshape(
            nphi, len(deltrs), ntheta, 3
        )
        outputHandler.saveNumpyData(emitter_locations, 'EmitterLocations_XYZ')
        outputHandler.log.info(
            'OUTPUT EMITTER LOCATIONS: EmitterLocations_XYZ.npy, shape={}'.format(
                emitter_locations.shape
            )
        )
    ## GENERATE INITIAL VELOCITIES
    initVel_array = generate_MB_velocities(N_particles=n_particles, normals_list=normals_list,
                                           ion_temp=temperature, ion_mass=mass,
                                           nparticles_per_emitter=nparticles_per_emitter, outputHandler=outputHandler)

    # Keep every initial-condition product in emitter-major order:
    # [emitter 0 particle 0..N, emitter 1 particle 0..N, ...].
    seed_array = np.asarray(seed_list, dtype=np.float64)
    normals_array = np.asarray(normals_list, dtype=np.float64)
    particle_positions = np.repeat(seed_array, nparticles_per_emitter, axis=0)
    particle_normals = np.repeat(normals_array, nparticles_per_emitter, axis=0)

    initVelPos = np.empty((n_particles, 6), dtype=np.float64)
    initVelPos[:, 0:3] = initVel_array
    initVelPos[:, 3:6] = particle_positions
    ion_list = [Ion(seed_pt, mass, charge) for seed_pt in particle_positions]

    ## SET INITIAL STATES AND OUTPUT(?necessary?)
    for ion, v_0 in zip(ion_list, initVel_array):
        ion.initVelocity(v_0)
        #ion.initOutput(DT, TMAX)

    if return_normals:
        return ion_list, initVelPos, particle_normals
    return ion_list, initVelPos
