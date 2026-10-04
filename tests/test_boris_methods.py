"""Small CPU checks; Warp tests skip when the optional dependency is absent."""
import importlib.util
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import illiad.boris as bm
import illiad.mesh.torch_mesh as tm
import illiad.utilities.coordtrans as ct
from illiad.particle import Ion


class BackendSelection(unittest.TestCase):
    def test_default_and_validation(self):
        solver = bm.Boris(None)
        self.assertEqual(solver.method, 'torch')
        self.assertEqual(bm.Boris.resolve_method(' WARP '), 'warp')
        for value in (None, '', 'cuda', False):
            with self.assertRaises(ValueError):
                bm.Boris(None, method=value)
        with patch.object(solver, 'torch_solver', return_value='torch') as dispatch:
            self.assertEqual(solver.parallel_solver([], None), 'torch')
            dispatch.assert_called_once()

    def test_missing_optional_dependency(self):
        if importlib.util.find_spec('warp') is None:
            with self.assertRaisesRegex(ImportError, r'illiad-fieldlines\[warp\]'):
                bm.Boris.require_method('warp')

    def test_default_torch_push(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(bm, 'device', torch.device('cpu')), \
                patch.object(tm, 'device', torch.device('cpu')), \
                patch.object(ct, 'device', torch.device('cpu')):
            path = Path(tmp) / 'b.npy'
            np.save(path, np.zeros((3, 3, 4, 4)))
            field = tm.TorchMesh(R0=.72, a=.19)
            field.loadCartesianField(str(path), att_mult=1.)
            ion = Ion(np.array([.8, 0., 0.]), 6.941, 1)
            ion.initVelocity(np.array([1., 0., 0.]))
            solver = bm.Boris(None)
            solver.setConditions([ion], 'test', dt=1e-8, tmax=1e-8)
            result = solver.parallel_solver([ion], field, trace_IDs=[0])
            np.testing.assert_allclose(result[-1][-1, 0], [.80000001, 0., 0.], atol=1e-12)


@unittest.skipUnless(importlib.util.find_spec('warp'), 'optional Warp is not installed')
class BackendParity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for module in (bm, tm, ct):
            patcher = patch.object(module, 'device', torch.device('cpu'))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.saved = {}
        self.io = SimpleNamespace(log=logging.getLogger('test'),
                                  saveNumpyData=lambda data, name: self.saved.update({name: data}))
        rng = np.random.default_rng(12)
        self.b = self.field(rng.normal(size=(3, 5, 8, 4)) * .03, 'b')
        self.e = self.field(rng.normal(size=(3, 5, 8, 20)) * 10, 'e', periods=1)
        path = Path(self.tmp.name) / 'density.npy'
        np.save(path, rng.uniform(.1, 1, (20, 8, 5)))
        self.n = tm.TorchMesh(R0=.72, a=.19)
        self.n.loadScalarField(str(path), att_mult=1e18)

    def field(self, data, name, periods=5):
        path = Path(self.tmp.name) / (name + '.npy')
        np.save(path, data)
        mesh = tm.TorchMesh(R0=.72, a=.19)
        mesh.loadCartesianField(str(path), att_mult=1.0, period_=np.array([0, 1, periods]))
        return mesh

    def particles(self, near_wall=False):
        r = .18999 if near_wall else .08
        angles = np.linspace(.1, 6.1, 7)
        points = ct.RTP_to_XYZ_many(np.column_stack([np.full(7, r), np.zeros(7), angles]), .72)
        ions = [Ion(p, 6.941, 1) for p in points]
        for ion, phi in zip(ions, angles):
            ion.initVelocity(np.array([np.cos(phi), -np.sin(phi), .01]) * 1e4)
        return ions

    def solver(self, ions, method, steps=7):
        solver = bm.Boris(self.io, method=method)
        solver.setConditions(ions, 'test', dt=1e-8, tmax=steps*1e-8)
        solver.nsteps = steps + 1
        return solver

    def test_deterministic_trace_and_wall_parity(self):
        for near_wall, stride, ids, freq in [(False, 3, [0, 4, -1], False),
                                             (True, 3, [0, 4], False),
                                             (False, 2, [], False),
                                             (False, 2, [2], True)]:
            ions = self.particles(near_wall)
            outputs = [self.solver(ions, method).parallel_solver(
                ions, self.b, self.e, trace_IDs=ids, trace_stride=stride,
                freq_corr=freq, ion_neutral_collisions='viscous_drag')
                for method in ('torch', 'warp')]
            for a, b in zip(*outputs):
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-10, atol=1e-8)
            expected_rows = 2 if near_wall else 1 + 7//stride + (7 % stride != 0)
            self.assertEqual(outputs[1][-1].shape, (expected_rows, len(ids), 3))

    def test_zero_field_frequency_correction(self):
        zero = self.field(np.zeros((3, 5, 8, 4)), 'zero')
        ions = self.particles()
        outputs = [self.solver(ions, method).parallel_solver(
            ions, zero, trace_IDs=[0], freq_corr=True) for method in ('torch', 'warp')]
        np.testing.assert_allclose(outputs[0][-1], outputs[1][-1], atol=1e-12)
        self.assertTrue(torch.isfinite(outputs[1][-1]).all())

    def test_run_output_and_collisions(self):
        for near_wall in (False, True):
            for model in ('linear_fp', 'fokker_planck'):
                ions = self.particles(near_wall)
                result = self.solver(ions, 'warp').run(
                    self.b, self.e, self.n, ion_neutral_collisions='langevin',
                    ion_ion_collisions=model, trace_IDs=[0, 2], trace_stride=3)
                self.assertEqual(result[0].shape[0], 7)
                self.assertLessEqual(result[0].shape[1], len(ions))
                if not near_wall:
                    self.assertEqual(result[0].shape[1], 0)
                self.assertTrue(np.isfinite(result[-1]).all())
                np.testing.assert_array_equal(self.saved['Wallpt_OUTPUT'], result[0])
                np.testing.assert_array_equal(self.saved['Ion_traces'], result[-1])

    def test_fields_share_storage(self):
        from illiad.boris_warp import _vector_grid_from_torch
        grid = _vector_grid_from_torch(self.b)
        self.assertEqual(grid.values.ptr, self.b.B.data_ptr())


if __name__ == '__main__':
    unittest.main()
