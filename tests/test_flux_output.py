"""Invalid fits must not prevent saving remaining surface geometry."""

import unittest
from unittest.mock import Mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from illiad.cli.flux_calc import DEFAULT_INPUTS
from illiad.flux.calculator import FluxCalculator


class FluxOutputTests(unittest.TestCase):
    def test_nonfinite_or_zero_flux_still_saves_surface_geometry(self):
        for values in ([.2, np.nan, np.inf], [np.nan]*3, [0.]*3):
            with self.subTest(values=values):
                io = Mock()
                calc = FluxCalculator(io, None, dict(DEFAULT_INPUTS, LCFS_INDEX=0,
                                      PHI_GENs=[360], NTHETA=4, BIG_MESH=True))
                calc.nsurfaces = 3
                calc.init_output_arrays()
                calc.tot_flux_array[:, 0] = values
                original = calc.tot_flux_array.copy()
                def check_plot(*args, **kwargs):
                    limits = plt.gcf().axes[0].get_ylim()
                    self.assertTrue(np.isfinite(limits).all())
                    self.assertLess(limits[0], limits[1])
                io.saveFig.side_effect = check_plot
                calc.save_output()
                np.testing.assert_array_equal(calc.tot_flux_array, original)
                paths = [call.args[1] for call in io.saveNumpyData.call_args_list]
                self.assertEqual(len(paths), 10)  # Four core arrays + centers/meshes.
                self.assertTrue(any(p.endswith('Iota.npy') for p in paths))
                self.assertTrue(any(p.endswith('fSurf_002_POINTmesh.npy') for p in paths))


if __name__ == '__main__':
    unittest.main()
