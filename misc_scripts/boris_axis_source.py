"""Boris transport from regularly spaced magnetic-axis emitters.

Run: python misc_scripts/boris_axis_source.py input_files/boris_axis_source.example.json
See docs/boris_axis_source.md for axis provenance and streaming histogram semantics.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import illiad.cli.boris as boris_cli
from illiad.particle import Ion
from illiad.utilities.coordtrans import RTP_to_XYZ_many
from illiad.utilities.physical_constants import KG_PER_AMU
from illiad.utilities.run_config import load_inputs_json, merge_input_params

DEFAULT_INPUTS_JSON = Path(__file__).resolve().parents[1] / 'input_files/boris_axis_source.example.json'
SOURCE_DEFAULTS = {
    'NPHI': 36,
    'AXIS_PHI_START_DEG': 0.0,
    'AXIS_FILE': None,
    'AXIS_PHI_DEG': None,
    'SOURCE_SEED': None,
    'TRACK_NPHI': 0,
    'TRACK_NPARTICLES_PER_EMITTER': 0,
    'PLANE_HISTOGRAM_PHI_DEG': None,
    'PLANE_HISTOGRAM_MIN_NORMAL_SPEED': 0.0,
}


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def load_axis_emitters(params, bfield):
    """Interpolate native (theta radians, rho metres) axis centers in (R-R0,Z).

    Native fSurf_NNN_center.npy axes use phi=dphi,...,360 degrees. An explicit
    AXIS_PHI_DEG list overrides that convention. Each plane must have exactly
    one center, with optional trailing zero/NaN padding from the native
    flux writer; island-center sets are not an axis trajectory.
    """
    if not params.get('AXIS_FILE'):
        raise ValueError('AXIS_FILE must select magnetic-axis centers for the chosen field configuration')
    axis = np.load(params['AXIS_FILE'], allow_pickle=False)
    if axis.ndim < 2 or axis.shape[-1] != 2 or len(axis) < 2:
        raise ValueError('AXIS_FILE must have shape (nphi, ..., 2) in (theta, rho) order')
    planes = axis.reshape(len(axis), -1, 2)
    centers = []
    for plane in planes:
        if not np.isfinite(plane[0]).all():
            raise ValueError('Each AXIS_FILE plane must start with a finite axis center')
        extra = plane[1:]
        if np.any(np.isfinite(extra).all(axis=1) & np.any(extra != 0, axis=1)):
            raise ValueError('AXIS_FILE contains multiple centers; select the magnetic-axis surface')
        centers.append(plane[0])
    centers = np.asarray(centers)
    if np.any((centers[:, 1] < 0) | (centers[:, 1] >= bfield.a)):
        raise ValueError('Magnetic-axis centers must lie inside the vessel')
    source_phi = params.get('AXIS_PHI_DEG')
    if source_phi is None:
        source_phi = np.linspace(360/len(axis), 360, len(axis))
    source_phi = np.asarray(source_phi, dtype=float)
    if (source_phi.shape != (len(axis),) or not np.isfinite(source_phi).all()
            or len(np.unique(source_phi % 360)) != len(axis)):
        raise ValueError('AXIS_PHI_DEG must give one finite, distinct periodic angle per axis plane')
    count = integer(params['NPHI'], 'NPHI')
    start = float(params['AXIS_PHI_START_DEG'])
    if not np.isfinite(start):
        raise ValueError('AXIS_PHI_START_DEG must be finite')
    phi = (start + np.arange(count)*360/count) % 360
    u = np.interp(phi, source_phi, centers[:, 1]*np.cos(centers[:, 0]), period=360)
    z = np.interp(phi, source_phi, centers[:, 1]*np.sin(centers[:, 0]), period=360)
    rtp = np.column_stack((np.hypot(u, z), np.arctan2(z, u), np.deg2rad(phi)))
    return RTP_to_XYZ_many(rtp, bfield.R0), phi


def generate_isotropic_velocities(n_particles, temperature_ev, mass_amu, rng):
    """Independent Cartesian Gaussian components: Maxwellian speeds, uniform solid angle."""
    if not np.isfinite(temperature_ev) or temperature_ev <= 0:
        raise ValueError('ION_TEMP must be finite and positive')
    if not np.isfinite(mass_amu) or mass_amu <= 0:
        raise ValueError('ION_MASS must be finite and positive')
    sigma = np.sqrt(1.602176634e-19*temperature_ev/(mass_amu*KG_PER_AMU))
    return rng.normal(0., sigma, (n_particles, 3))


def make_axis_initializer(params):
    def initialize(initial_conditions, ion_properties, bfield, efield, outputHandler):
        del efield
        mass, charge, temperature = ion_properties
        emitters, phi = load_axis_emitters(params, bfield)
        copies = integer(initial_conditions[-1], 'NPARTICLES_PER_EMITTER')
        positions = np.repeat(emitters, copies, axis=0)
        velocities = generate_isotropic_velocities(
            len(positions), float(temperature), float(mass), np.random.default_rng(params['SOURCE_SEED']))
        ions = [Ion(position, mass, charge) for position in positions]
        for ion, velocity in zip(ions, velocities):
            ion.initVelocity(velocity)
        # Reference +phi direction for angular diagnostics only; no launch bias.
        reference = np.column_stack((-np.sin(np.deg2rad(phi)), -np.cos(np.deg2rad(phi)), np.zeros(len(phi))))
        outputHandler.saveNumpyData(emitters, 'AxisIonSeedPts')
        outputHandler.saveNumpyData(phi, 'AxisIonSeedPhi_deg')
        settings_path = Path(outputHandler._outputPath(outputHandler.data_dir, 'axis_source_settings.json'))
        settings_path.write_text(json.dumps(params, indent=2)+'\n')
        outputHandler.log.info('MAGNETIC AXIS SOURCE: %s; %d emitters, %d particles/emitter; '
                               'isotropic Maxwellian, T=%g eV, phi start=%g deg, spacing=%g deg',
                               params['AXIS_FILE'], len(emitters), copies, temperature,
                               params['AXIS_PHI_START_DEG'], 360/len(emitters))
        return ions, np.column_stack((velocities, positions)), np.repeat(reference, copies, axis=0)
    return initialize


def prepare_params(overrides):
    params = merge_input_params(boris_cli.DEFAULT_INPUTS, SOURCE_DEFAULTS)
    params = merge_input_params(params, overrides)
    for key in ('NPHI', 'NPARTICLES_PER_EMITTER', 'STRIDE'):
        params[key] = integer(params[key], key)
    for track, total in (('TRACK_NPHI', 'NPHI'), ('TRACK_NPARTICLES_PER_EMITTER', 'NPARTICLES_PER_EMITTER')):
        params[track] = integer(params[track], track, minimum=0)
        if params[track] > params[total]:
            raise ValueError(f'{track} cannot exceed {total}')
    params.update(NTHETA=1, TRACK_NTHETA=1, DELTRS=[0.0], LCFS_INDEX=0)
    params['TAG'] = 'axis_' + str(params['TAG'])
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs_path', nargs='?')
    parser.add_argument('--inputs', '--inputs-json', dest='inputs_json')
    args = parser.parse_args(argv)
    if args.inputs_path and args.inputs_json:
        parser.error('provide INPUTS or --inputs, not both')
    params = prepare_params(load_inputs_json(args.inputs_json or args.inputs_path or DEFAULT_INPUTS_JSON,
                                            'Magnetic-axis Boris inputs'))
    boris_cli.boris_runner(params, initializer=make_axis_initializer(params), velocity_distribution='isotropic')


if __name__ == '__main__':
    main()
