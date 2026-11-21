"""Generate straight-angle coordinate plots without rerunning flux integration."""

import argparse
import logging
import matplotlib
matplotlib.use('Agg')

from illiad.cli.flux_calc import DEFAULT_INPUTS
from illiad.flux.calculator import FluxCalculator
from illiad.io import IOHandler
from illiad.mesh import Mesh
from illiad.utilities.run_config import load_inputs_json, merge_input_params, normalize_phi_gens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--n-theta', type=int, help='Number of constant straight-angle lines')
    parser.add_argument('--planes', type=float, nargs='+')
    args = parser.parse_args()
    params = merge_input_params(DEFAULT_INPUTS, load_inputs_json(args.inputs, 'Flux calculator inputs'))
    params['PLOT_STRAIGHT_ANGLE'] = True
    if args.n_theta is not None:
        params['SL_NTHETA'] = args.n_theta
    if args.planes is not None:
        params['PHI_GENs'] = args.planes
    normalize_phi_gens(params)
    logging.basicConfig(level=logging.INFO)
    io = IOHandler(params['ANLYS_DIR'])
    io.setActiveSubDir(params['ANLYS_SUBDIR'])
    calc = FluxCalculator(io, Mesh(R0=.72, a=.19), params)
    for phi in calc.phi_gens:
        surfaces = calc.load_poincare_data(phi)
        calc.nsurfaces = len(surfaces)
        axis = calc.find_axis(*surfaces[-1].copy(), calc.field)
        counts = calc.identify_surface_subsets(surfaces, axis)[1]
        calc.straight_angle_plots.plot_plane(phi, surfaces, axis, counts)


if __name__ == '__main__':
    main()
