"""Replot surface parameters using saved fluxes and replayed geometric fits."""
import argparse
import numpy as np
from illiad.cli.flux_calc import DEFAULT_INPUTS
from illiad.flux.calculator import FluxCalculator
from illiad.flux.surface_parameter_plots import plot_parameters
from illiad.io import IOHandler
from illiad.mesh import Mesh
from illiad.utilities.run_config import load_inputs_json, merge_input_params, normalize_phi_gens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    args = parser.parse_args()
    params = merge_input_params(DEFAULT_INPUTS, load_inputs_json(args.inputs, 'Flux calculator inputs'))
    normalize_phi_gens(params)
    params.update(PLOT_STRAIGHT_ANGLE=False, PLOT_SURFACE_PARAMETER=True)
    io = IOHandler(params['ANLYS_DIR'])
    io.setActiveSubDir(params['ANLYS_SUBDIR'])
    calc = FluxCalculator(io, Mesh(R0=.72, a=.19), params)
    for plane, phi in enumerate(calc.phi_gens):
        surfaces = calc.load_poincare_data(phi)
        calc.nsurfaces = len(surfaces)
        if plane == 0:
            calc.init_output_arrays()
            calc.total_flux_norm[:] = np.nan
            if params['SURFACE_PARAMETER'] in ('flux', 'both', 'blend', 'all') or params['PLOT_PARAMETER_PROFILES']:
                flux = io.loadNumpyData('CalculatedFLuxes-normalized.npy', subdir=params['ANLYS_SUBDIR'])
                if flux.shape != calc.total_flux_norm.shape:
                    raise ValueError('Saved flux shape differs from selected surfaces/phi planes')
                calc.total_flux_norm = flux
        axis = calc.find_axis(*surfaces[-1].copy(), calc.field)
        smallest, counts, subsets, centers, iota = calc.identify_surface_subsets(surfaces, axis)
        calc.iota_array[:, plane] = iota[0]
        reverse = axis.copy()
        reverse[0] += np.pi
        lcfs = None
        for surface in range(calc.lcfs_index, calc.nsurfaces):
            geo, magnetic, _, _, valid = calc.fit_and_integrate_surface_subsets(
                subsets, centers, surface, smallest, int(counts[surface]), iota[-1],
                reverse, phi, integrate=False)
            calc.valid_surfs[surface, plane] = valid
            if surface == calc.lcfs_index:
                lcfs = magnetic
            else:
                calc.surface_validation(surface, plane, magnetic, int(counts[surface]), lcfs)
            if calc.valid_surfs[surface, plane]:
                theta, radius = geo[..., 0], geo[..., 1]
                calc.parameter_contours[plane][surface] = np.stack(
                    (radius*np.cos(theta), radius*np.sin(theta)), axis=-1)
    plot_parameters(calc)


if __name__ == '__main__':
    main()
