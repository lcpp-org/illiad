"""Analytic geometry, trace termination, identity, IO and backend checks."""
import importlib.util
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from illiad.boundaries import (Box, BoundaryRecorder, parse_boundaries, segment_hits,
                               torch_box_hits, validate_initial_positions, iter_events)
from illiad.boundary_fieldlines import trace_fieldline


def box(center=(0, 0, 0), size=(2, 2, 2), angles=(0, 0, 0), mode='transparent', name='box'):
    return Box.from_config(dict(name=name, center_m=center, width_m=size[0], depth_m=size[1],
                               height_m=size[2], rotation_local_xyz_deg=angles, mode=mode))


class Geometry(unittest.TestCase):
    def test_rzphi_center_convention_and_legacy_roundtrip(self):
        from illiad.utilities.coordtrans import RZphi_to_XYZ, XYZ_to_RZphi, RTP_to_XYZ
        config = box().config()
        config['center_rzphi'] = [.72, -.12, 90.]
        placed = Box.from_config(config)
        np.testing.assert_allclose(placed.center, [0., -.72, -.12], atol=1e-15)
        config['center_rzphi'][2] = -270.
        np.testing.assert_allclose(Box.from_config(config).center, placed.center, atol=1e-15)
        rho, theta, phi = .1, .7, np.deg2rad(306.)
        rzphi = [.72+rho*np.cos(theta), rho*np.sin(theta), 306.]
        np.testing.assert_allclose(RZphi_to_XYZ(rzphi), RTP_to_XYZ([rho, theta, phi]), atol=1e-15)
        np.testing.assert_allclose(XYZ_to_RZphi(RZphi_to_XYZ(rzphi)), rzphi, atol=1e-13)
        legacy = box((.2, -.4, .3), angles=(21, 37, 63))
        restored = Box.from_config(legacy.config())
        np.testing.assert_allclose(restored.center, legacy.center, atol=1e-15)
        np.testing.assert_array_equal(restored.rotation, legacy.rotation)
        with self.assertRaises(ValueError):
            Box.from_config(dict(config, center_m=[0, 0, 0]))
        for invalid in ([-1, 0, 0], [.72, 0], [.72, 0, float('nan')]):
            with self.assertRaises(ValueError):
                Box.from_config(dict(config, center_rzphi=invalid))

    def test_rotation_and_crossings(self):
        b = box(center=(.2, -.4, .3), angles=(21, 37, 63))
        for p, q, expected in [([-2, 0, 0], [2, 0, 0], [(0.25, -1), (.75, 1)]),
                               ([0, 0, 0], [2, 0, 0], [(.5, 1)]),
                               ([-2, 2, 0], [2, 2, 0], []),
                               ([-2, 0, 0], [-1, 0, 0], [(1., -1)]),
                               ([-1, 0, 0], [0, 0, 0], []),
                               ([-2, 1, 0], [2, 1, 0], [(.25, 0), (.75, 0)]),
                               ([-2, 0, 0], [0, 2, 0], [(.5, 0)])]:
            start, end = b.center+np.array(p)@b.rotation.T, b.center+np.array(q)@b.rotation.T
            actual = segment_hits(b, start, end)
            np.testing.assert_allclose(actual, expected, atol=1e-13)
            roots, kinds = torch_box_hits(b, torch.tensor(start[None]), torch.tensor(end[None]))
            valid = torch.isfinite(roots)
            np.testing.assert_allclose(roots[valid], [v[0] for v in expected], atol=1e-13)
            np.testing.assert_array_equal(kinds.expand_as(roots)[valid], [v[1] for v in expected])
        r = box(angles=(90, 90, 0)).rotation
        np.testing.assert_allclose(r @ [0, 0, 1], [1, 0, 0], atol=1e-15)

    def test_validation_and_corner(self):
        with self.assertRaises(ValueError):
            parse_boundaries([box(), box()])
        with self.assertRaises(ValueError):
            box(size=(-1, 1, 1))
        with self.assertRaises(ValueError):
            validate_initial_positions([box(mode='absorbing')], [[0, 0, 0]])
        recorder = BoundaryRecorder([box()])
        recorder.record(0, 17, 22, 1, 4, .5, .2, [[1, 1, 1]], [[1, 1, 1]], 1)
        row = recorder.events()[0]
        self.assertEqual(row['face_mask'], 2+8+32)
        self.assertEqual(row['face_id'], 1)
        np.testing.assert_array_equal(row['uv_m'], [1, 1])

    def test_event_order_wall_and_absorber(self):
        boxes = [box((.82, 0, 0), (.01, .1, .1), name='near'),
                 box((.85, 0, 0), (.01, .1, .1), mode='absorbing', name='stop'),
                 box((.89, 0, 0), (.01, .1, .1), name='far')]
        rec = BoundaryRecorder(boxes)
        end, stop, fraction = rec.process_torch(
            torch.tensor([[.8, 0., 0.]]), torch.tensor([[.95, 0., 0.]]),
            torch.tensor([8]), 1, 0., 1., torch.tensor([[.15, 0., 0.]]),
            major_radius=.72, vessel_radius=.19)
        np.testing.assert_allclose(end, [[.845, 0, 0]], atol=1e-7)
        self.assertTrue(stop.item())
        events = rec.events()
        np.testing.assert_array_equal(events['boundary_id'], [0, 0, 1])
        rec = BoundaryRecorder([box((.925, 0, 0), (.01, .1, .1))])
        rec.process_torch(torch.tensor([[.8, 0., 0.]]), torch.tensor([[.95, 0., 0.]]),
                          torch.tensor([0]), 1, 0., 1., torch.ones((1, 3)), major_radius=.72, vessel_radius=.19)
        self.assertEqual(len(rec.events()), 0)

    def test_vessel_and_coincident_box_ties(self):
        stop = box((.92, 0, 0), (.02, .1, .1), mode='absorbing')
        rec = BoundaryRecorder([stop])
        _, absorbed, _ = rec.process_torch(
            torch.tensor([[.9, 0., 0.]], dtype=torch.float64),
            torch.tensor([[.93, 0., 0.]], dtype=torch.float64), torch.tensor([0]), 1, 0., 1.,
            torch.ones((1, 3), dtype=torch.float64), major_radius=.72, vessel_radius=.19)
        self.assertFalse(absorbed.item())
        self.assertEqual(len(rec.events()), 0)
        rec = BoundaryRecorder([box((.82, 0, 0), (.01, .1, .1), mode='absorbing', name='first'),
                                box((.82, 0, 0), (.01, .1, .1), mode='absorbing', name='second')])
        rec.process_torch(torch.tensor([[.8, 0., 0.]], dtype=torch.float64),
                          torch.tensor([[.84, 0., 0.]], dtype=torch.float64), torch.tensor([0]),
                          1, 0., 1., torch.ones((1, 3), dtype=torch.float64), major_radius=.72, vessel_radius=.19)
        np.testing.assert_array_equal(rec.events()['boundary_id'], [0])

    def test_stream_archive_and_plot(self):
        from misc_scripts.plot_boundary_events import plot_events
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = BoundaryRecorder([box()], root/'events', buffer_rows=2)
            for i in range(5):
                rec.record(0, i, i+1, 0, 1, .5, .5, [[1, 0, 0]], [[2, 0, 0]], 1, 1e-27)
            rec.set_vessel(.72, .19)
            rec.finish()
            meta = json.loads((root/'events/manifest.json').read_text())
            self.assertEqual(meta['vessel']['minor_radius_m'], .19)
            self.assertIn('center_rzphi', meta['boxes'][0])
            self.assertEqual(sum(len(a) for a in iter_events(root/'events')), 5)
            self.assertTrue(json.loads((root/'events/manifest.json').read_text())['complete'])
            plot_events(root/'events', root/'plots', bins=4)
            self.assertTrue((root/'plots/boundary_preview.png').exists())
            with np.load(root/'plots/box_000_count_all.npz') as data:
                self.assertEqual(data['values'].sum(), 5)
            for metric, expected in [('area', 20), ('unique', 5), ('energy', 2e-27/1.602176634e-19), ('angle', 0)]:
                plot_events(root/'events', root/'plots', bins=4, metric=metric)
                with np.load(root/f'plots/box_000_{metric}_all.npz') as data:
                    self.assertAlmostEqual(np.nansum(data['values']), expected)


class _StraightField:
    R0, a = .72, .19

    def interpField(self, xyz):
        return np.array([[1., 0., 0.]])


class _PoincareIO:
    def __init__(self, directory):
        self.data_dir = str(directory)
        self.log = logging.getLogger('poincare-boundary-test')

    def createSubDir(self, name):
        (Path(self.data_dir)/name).mkdir(exist_ok=True)

    def _outputPath(self, base, name):
        return str(Path(base)/name)


class PoincareWorkflow(unittest.TestCase):
    def test_parallel_workers_and_merged_manifest(self):
        from illiad.poincare import Poincare
        with tempfile.TemporaryDirectory() as tmp:
            solver = Poincare(_PoincareIO(tmp), solvr='LSODA', workers=1, double_line=True)
            solver.set_conditions(np.array([[.08, 0., 0.]]), spins=.04/(2*np.pi*.72),
                                  field=_StraightField(), nplanes=4)
            solver.set_boundaries([box((.82, 0, 0), (.01, .1, .1), mode='absorbing')])
            results = list(solver.parallel_solver())
            np.testing.assert_allclose([r[0] for r in results], [.015, .04], atol=1e-10)
            self.assertEqual(sum(len(r[2]) for r in results), 0)
            # Plot workers are outside this integration/manifest test.
            with patch('illiad.poincare.cf.ProcessPoolExecutor'):
                output = solver.post_solver(iter(results))
            self.assertAlmostEqual(output[0][0], .055)
            events = np.concatenate(list(iter_events(Path(tmp)/'Boundary_events')))
            self.assertEqual(len(events), 1)
            self.assertEqual(events['trajectory_id'][0], 0)
            self.assertEqual(events['direction'][0], 1)


class FieldLines(unittest.TestCase):
    def test_adaptive_transparent_and_absorbing(self):
        for mode, expected in [('transparent', 2), ('absorbing', 1)]:
            rec = BoundaryRecorder([box((.82, 0, 0), (.01, .1, .1), mode=mode)], coordinate='arc_length_m')
            result = trace_fieldline(lambda s, x: np.array([1., 0, 0]), [.8, 0, 0], .1, rec,
                                     method='RK45', max_step=.1)
            self.assertEqual(len(rec.events()), expected)
            np.testing.assert_allclose(rec.events()['coordinate'][0], .015, atol=1e-12)
            self.assertAlmostEqual(result.t[-1], .015 if mode == 'absorbing' else .1)
            self.assertEqual(len(result.y_events[0]), 0)
        empty = BoundaryRecorder([])
        ref = trace_fieldline(lambda s, x: np.array([1., 0, 0]), [.8, 0, 0], .1, empty, method='RK45', max_step=.1)
        trans = BoundaryRecorder([box((.82, 0, 0), (.01, .1, .1))])
        actual = trace_fieldline(lambda s, x: np.array([1., 0, 0]), [.8, 0, 0], .1, trans, method='RK45', max_step=.1)
        np.testing.assert_array_equal(actual.y, ref.y)

    def test_coincident_transparent_and_absorbing(self):
        rec = BoundaryRecorder([box((.82, 0, 0), (.01, .1, .1), mode='absorbing', name='stop'),
                                box((.82, 0, 0), (.01, .1, .1), name='diagnostic')], coordinate='arc_length_m')
        result = trace_fieldline(lambda s, x: np.array([1., 0, 0]), [.8, 0, 0], .1, rec)
        np.testing.assert_array_equal(rec.events()['boundary_id'], [0, 1])
        self.assertAlmostEqual(result.t[-1], .015)

    def test_curved_and_reverse(self):
        # Circle around (.72, 0, 0) in XZ plane: first z=.03 at s=.05*asin(.6).
        b = box((.76, 0, .035), (.1, .1, .01), mode='absorbing')
        errors = []
        for max_step in (.02, .01):
            rec = BoundaryRecorder([b], coordinate='arc_length_m')
            rhs = lambda s, x: np.array([-x[2]/.05, 0, (x[0]-.72)/.05])
            result = trace_fieldline(rhs, [.77, 0, 0], .1, rec, method='RK45',
                                     max_step=max_step, rtol=1e-10, atol=1e-12)
            errors.append(abs(result.t[-1]-.05*np.arcsin(.6)))
        self.assertLess(max(errors), 1e-9)
        rec = BoundaryRecorder([box((.78, 0, 0), (.01, .1, .1))], coordinate='arc_length_m')
        trace_fieldline(lambda s, x: np.array([-1., 0, 0]), [.8, 0, 0], .05, rec, direction=-1)
        np.testing.assert_array_equal(rec.events()['direction'], [-1, -1])


class BorisBoundary(unittest.TestCase):
    def setUp(self):
        import illiad.boris as bm
        import illiad.mesh.torch_mesh as tm
        import illiad.utilities.coordtrans as ct
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        device = torch.device(os.environ.get('ILLIAD_BOUNDARY_TEST_DEVICE', 'cpu'))
        for module in (bm, tm, ct):
            patcher = patch.object(module, 'device', device)
            patcher.start(); self.addCleanup(patcher.stop)
        self.bm, self.tm = bm, tm
        np.save(self.root/'b.npy', np.zeros((3, 5, 8, 4)))
        self.field = tm.TorchMesh(R0=.72, a=.19)
        self.field.loadCartesianField(str(self.root/'b.npy'), att_mult=1.)

    def run_case(self, method, mode, chunk=4, compact=4, capacity=1024, angles=(0, 0, 0)):
        from illiad.particle import Ion
        starts = [[.8, 0, 0], [.8, .04, 0], [.9, 0, 0]]
        ions = [Ion(np.array(p), 6.941, 1) for p in starts]
        for ion in ions:
            ion.initVelocity(np.array([1e4, 0, 0]))
        solver = self.bm.Boris(None, method=method)
        solver.setConditions(ions, 'test', dt=1e-6, tmax=10e-6)
        solver.nsteps = 11
        rec = None if mode is None else BoundaryRecorder(
            [box((.825, 0, 0), (.004, .02, .02), mode=mode, angles=angles)], buffer_rows=capacity)
        result = solver.parallel_solver(ions, self.field, trace_IDs=[0, 1, 2, 0], trace_stride=1,
                                        warp_step_chunk_size=chunk, warp_compaction_interval=compact,
                                        boundary_recorder=rec)
        return [a.cpu().numpy() for a in result], rec

    def test_torch_stop_and_transparent_invariance(self):
        ref, _ = self.run_case('torch', None)
        trans, rec = self.run_case('torch', 'transparent')
        for a, b in zip(ref, trans):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(len(rec.events()), 2)
        result, rec = self.run_case('torch', 'absorbing')
        np.testing.assert_allclose(result[-1][-1, 0], [.823, 0, 0], atol=1e-12)
        np.testing.assert_array_equal(result[0][0], [0, 0, 0])
        self.assertEqual(len(rec.events()), 1)
        self.assertEqual(rec.events()['trajectory_id'][0], 0)
        self.assertAlmostEqual(rec.events()['coordinate'][0], 2.3e-6)

    @unittest.skipUnless(importlib.util.find_spec('warp'), 'optional Warp unavailable')
    def test_warp_parity_chunking_and_compaction(self):
        for mode in ('absorbing', 'transparent'):
            reference, records = self.run_case('torch', mode)
            for chunk, compact in ((1, 0), (4, 4), (16, 256)):
                result, rec = self.run_case('warp', mode, chunk, compact)
                for a, b in zip(result, reference):
                    np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
                expected, actual = records.events(), rec.events()
                for name in expected.dtype.names:
                    if name != 'source_id':
                        np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=1e-12)
        for mode in ('absorbing', 'transparent'):
            reference, records = self.run_case('torch', mode, angles=(15, 30, 60))
            result, rec = self.run_case('warp', mode, angles=(15, 30, 60))
            for a, b in zip(result, reference):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(rec.events()['xyz_m'], records.events()['xyz_m'], atol=1e-12)
            np.testing.assert_array_equal(rec.events()['face_id'], records.events()['face_id'])
        with self.assertRaisesRegex(RuntimeError, 'overflow'):
            self.run_case('warp', 'transparent', capacity=1)


class SOLBoundaries(unittest.TestCase):
    def test_directional_batch_metadata_and_frozen_endpoints(self):
        from illiad.sol.tracer import integrate_directional_batch
        import illiad.sol.tracer as tracer
        field = SimpleNamespace(B=torch.zeros(1), R0=.72, a=.19)
        io = SimpleNamespace(log=logging.getLogger('boundary-sol-test'))
        store = SimpleNamespace(n_planes=4, count=0, append=lambda *args: None)
        def straight(field, positions, direction, step, integrator, minimum):
            candidate = positions.clone()
            candidate[:, 0] += direction*step
            return candidate, torch.ones(len(positions), dtype=torch.bool)
        for mode in ('transparent', 'absorbing'):
            rec = BoundaryRecorder([box((.82, 0, 0), (.01, .1, .1), mode=mode)], coordinate='arc_length_m')
            with patch.object(tracer, 'fieldline_step', side_effect=straight):
                result = integrate_directional_batch(
                    np.array([[.8, 0, 0], [.8, 0, 0]]), np.array([0., 0.]), [5, 5], [0, 1], [1., -1.],
                    field, .04, .01, store, io, 1, 1, False, 8, False, 'rk4', 1e-12, 40, 100, 100,
                    boundary_recorder=rec)
            np.testing.assert_array_equal(result['hit_wall'], [False, False])
            np.testing.assert_array_equal(result['hit_boundary'], [mode == 'absorbing', False])
            np.testing.assert_allclose(result['path_length'], [.015 if mode == 'absorbing' else .04, .04])
            if mode == 'absorbing':
                np.testing.assert_allclose(result['boundary_xyz'][0], [.815, 0, 0])
            self.assertTrue(np.all(result['valid_trace']))
            self.assertEqual(len(rec.events()), 1 if mode == 'absorbing' else 2)

    def test_volume_output(self):
        import illiad.sol.tracer as tracer
        field = SimpleNamespace(B=torch.zeros(1), R0=.72, a=.19)
        store = SimpleNamespace(n_planes=4, count=0, counts=np.zeros(4), append=lambda *args: None, finish=lambda: None)
        def straight(field, positions, direction, step, integrator, minimum):
            candidate = positions.clone(); candidate[:, 0] += direction*step
            return candidate, torch.ones(len(positions), dtype=torch.bool)
        with tempfile.TemporaryDirectory() as tmp:
            io = SimpleNamespace(log=logging.getLogger('boundary-sol-test'), data_dir=tmp)
            settings = dict(BOUNDARIES=[box((.82, 0, 0), (.01, .1, .1), mode='absorbing').config()],
                            ANLYS_SUBDIR='test', SPINS=.04/(2*np.pi*.72), LCFS_INDEX=0, VESSEL_RADIUS_M=.19)
            with patch.object(tracer, 'fieldline_step', side_effect=straight):
                result = tracer.trace_connection_length_volume(
                    {'initial_conditions_rtp': np.array([[.08, 0, 0]])}, settings, field, .01, 4,
                    store, io, False, 8, False, 'rk4', 1e-12, 40, 100, 100)
            np.testing.assert_allclose(result['direction_length'], [[.015, .04]])
            np.testing.assert_array_equal(result['hit_boundary'], [[True, False]])
            self.assertTrue((Path(tmp)/'test/Boundary_events/manifest.json').exists())
            settings['ANLYS_SUBDIR'] = 'paired'
            settings['N_PLANES'] = 4
            seed_data = dict(initial_conditions_rtp=np.array([[.08, 0, 0]]),
                             seed_plane_index=np.array([0]), seed_id=np.array([0]))
            from unittest.mock import Mock
            sink = Mock()
            with patch.object(tracer, 'fieldline_step', side_effect=straight):
                paired = tracer.trace_connection_length_volume_paired(
                    seed_data, settings, field, .01, 4, 100, lambda n: sink, Mock(), 100,
                    io, False, 8, False, 'rk4', 1e-12, 40, 100, 100)
            np.testing.assert_allclose(paired['direction_length'], result['direction_length'])
            np.testing.assert_array_equal(paired['hit_boundary'], result['hit_boundary'])
            sink.consume.assert_called_once()
            settings['BOUNDARIES'] = []
            with patch.object(tracer, 'fieldline_step', side_effect=straight):
                unchanged = tracer.trace_connection_length_volume(
                    seed_data, settings, field, .01, 4, store, io, False, 8, False,
                    'rk4', 1e-12, 40, 100, 100)
            np.testing.assert_allclose(unchanged['direction_length'], [[.04, .04]])
            self.assertNotIn('boundary_xyz', unchanged)


if __name__ == '__main__':
    unittest.main()
