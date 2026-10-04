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

    def test_chunk_size_validation(self):
        for value in (0, -1, True, np.bool_(True), 1.5, 2.0, '16', None, 2**31):
            with self.assertRaisesRegex(ValueError, 'WARP_STEP_CHUNK_SIZE'):
                bm.Boris.validate_warp_step_chunk_size(value)
        self.assertEqual(bm.Boris.validate_warp_step_chunk_size(np.int64(16)), 16)

    def test_compaction_interval_validation(self):
        for value in (-1, True, np.bool_(True), 1.5, 2.0, '256', None, 2**31):
            with self.assertRaisesRegex(ValueError, 'WARP_COMPACTION_INTERVAL'):
                bm.Boris.validate_warp_compaction_interval(value)
        for value in (0, 1, np.int64(256)):
            self.assertEqual(bm.Boris.validate_warp_compaction_interval(value), value)

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
    @classmethod
    def setUpClass(cls):
        import warp as wp
        wp.init()

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
                self.assertEqual(self.saved['Ion_traces'].dtype, np.float32)
                np.testing.assert_array_equal(self.saved['Ion_traces'], result[-1].astype(np.float32))

    def test_fields_share_storage(self):
        from illiad.boris_warp import _vector_grid_from_torch
        grid = _vector_grid_from_torch(self.b)
        self.assertEqual(grid.values.ptr, self.b.B.data_ptr())

    def test_chunk_trace_boundaries(self):
        from illiad import boris_warp as wb
        grid = wb.make_grid(np.zeros((3, 4, 4, 3)), R0=.72, a=.19, periods=5, device='cpu')
        # Analytic radial paths hit on steps 1, 4, 8, 17; the last survives.
        x = np.column_stack([.72 + .19 - np.array([.5, 3.5, 7.5, 16.5, 40])*1e-4,
                             np.zeros(5), np.zeros(5)])
        v = np.tile([1e4, 0., 0.], (5, 1))
        for count in (4, 5):
            ids = [count-1, 0, 2, 2, -count]
            for steps in ((0, 19, 600) if count == 4 else (0, 19)):
                for stride in (1, 3, 8, 64):
                    final_step = min(steps, 17) if count == 4 else steps
                    samples = list(range(0, final_step + 1, stride))
                    if samples[-1] != final_step:
                        samples.append(final_step)
                    expected = np.stack([x[:count] + v[:count]*1e-8*np.minimum(
                        step, np.array([1, 4, 8, 17, 40])[:count])[:, None] for step in samples])[:, ids]
                    for chunk in (1, 2, 4, 16, 64):
                        with self.subTest(count=count, steps=steps, stride=stride, chunk=chunk):
                            result = wb._run_particles(x[:count], v[:count], 1., grid,
                                dt=1e-8, steps=steps, trace_ids=ids, trace_stride=stride,
                                step_chunk_size=chunk)
                            trace = result['traces'].numpy()[:result['trace_count']]
                            np.testing.assert_allclose(trace, expected, atol=1e-12, rtol=0)
                            hits = np.array([1, 4, 8, 17, -1])[:count] if steps else np.full(count, -1)
                            np.testing.assert_array_equal(result['hit_step'].numpy(), hits)

    def test_chunk_collision_rng_parity(self):
        from illiad import boris_warp as wb
        grid = wb._vector_grid_from_torch(self.b)
        electric = wb._vector_grid_from_torch(self.e)
        density = wb.DensityGrid()
        density.values = wb.wp.from_torch(self.n.value, dtype=wb.wp.float64)
        ions = self.particles() + self.particles(True)
        x = np.array([ion.pos0_XYZ for ion in ions])
        v = np.array([ion.vel0_XYZ for ion in ions])
        q = np.array([ion.charge_mass_ratio for ion in ions])
        for neutral, ion in [('viscous_drag', None), ('langevin', None),
                             ('langevin', 'linear_fp'), ('langevin', 'fokker_planck')]:
            config = wb.make_collision_config(ion_neutral_collisions=neutral, ion_ion_collisions=ion)
            reference = None
            for chunk, interval in ((c, i) for c in (1, 4, 16, 128) for i in (0, 4, 256)):
                result = wb._run_particles(x, v, q, grid, dt=1e-8, steps=35,
                    e_grid=electric, density_grid=density, collisions=config, seed=317,
                    trace_ids=list(range(len(ions))), step_chunk_size=chunk,
                    compaction_interval=interval)
                arrays = {key: result[key].numpy() for key in
                          ('wall_position_xyz', 'wall_velocity_xyz', 'hit_step', 'last_inside_step')}
                arrays['traces'] = result['traces'].numpy()[:result['trace_count']]
                if reference is None:
                    reference = arrays
                else:
                    for key in arrays:
                        np.testing.assert_allclose(arrays[key], reference[key], rtol=1e-12, atol=1e-10,
                                                   err_msg=f'{neutral}/{ion}, chunk={chunk}, compact={interval}, {key}')

    def test_chunk_launch_count(self):
        from illiad import boris_warp as wb
        ions = self.particles()
        with patch.object(wb.wp, 'launch', wraps=wb.wp.launch) as launch:
            self.solver(ions, 'warp', steps=19).run(
                self.b, self.e, trace_IDs=[0], trace_stride=1, warp_step_chunk_size=4)
        pushes = [call for call in launch.call_args_list if call.args[0] is wb.push_and_record_wall]
        self.assertEqual(len(pushes), 5)

    def test_compaction_launch_sizes_and_trace_identity(self):
        from illiad import boris_warp as wb
        zero = self.field(np.zeros((3, 5, 8, 4)), 'zero')
        ions = [Ion(np.array([r, 0., 0.]), 6.941, 1) for r in (.90995, .90955, .8)]
        for ion in ions:
            ion.initVelocity(np.array([1e4, 0., 0.]))
        # First two hit on steps 1 and 5; non-divisible interval rounds up to
        # step 6. Keep duplicate/negative trace selections across ID filtering.
        reference = None
        for interval in (0, 4):
            with patch.object(wb.wp, 'launch', wraps=wb.wp.launch) as launch:
                result = self.solver(ions, 'warp', steps=19).run(
                    zero, trace_IDs=[2, 0, -2, 0], trace_stride=1,
                    warp_step_chunk_size=3, warp_compaction_interval=interval)
            pushes = [c.kwargs['dim'] for c in launch.call_args_list
                      if c.args[0] is wb.push_and_record_wall]
            self.assertEqual(pushes, [3]*7 if interval == 0 else [3, 3, 1, 1, 1, 1, 1])
            if reference is None:
                reference = result
            else:
                for actual, expected in zip(result, reference):
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        with patch.object(wb.wp, 'launch', wraps=wb.wp.launch) as launch:
            result = self.solver(ions[:2], 'warp', steps=19).parallel_solver(
                ions[:2], zero, trace_IDs=[0, 1], trace_stride=3,
                warp_step_chunk_size=3, warp_compaction_interval=4)
        pushes = [c.kwargs['dim'] for c in launch.call_args_list
                  if c.args[0] is wb.push_and_record_wall]
        self.assertEqual(pushes, [2, 2])  # All terminated: no zero-sized push.
        self.assertEqual(tuple(result[-1].shape), (3, 2, 3))  # steps 0, 3, 5


if __name__ == '__main__':
    unittest.main()
