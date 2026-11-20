"""Streaming Boris crossings on cells bounded by native field-mesh nodes."""
from pathlib import Path

import numpy as np
import torch


class PlaneCrossingHistogram:
    """Combined event counts, independent of particle count and trace stride.

    Computational phi is -atan2(y, x). Segments are (start, end]: an
    initial point on the plane is not a crossing. Only the positive-R half
    plane and crossings inside the vessel are counted. Positions within
    1e-14 m of the plane are snapped to it to stabilize exact endpoints.
    """

    plane_tolerance_m = 1e-14

    def __init__(self, mesh, phi_deg, min_normal_speed=0.):
        if not np.isfinite(phi_deg):
            raise ValueError('PLANE_HISTOGRAM_PHI_DEG must be finite or null')
        if tuple(mesh.periodicity[:2]) != (0, 1) or mesh.nr < 2 or mesh.ntheta < 2:
            raise ValueError('Plane bins require a nonperiodic rho and full periodic theta mesh')
        self.phi_deg = float(phi_deg) % 360.0
        self.sin_phi, self.cos_phi = np.sin(np.deg2rad(self.phi_deg)), np.cos(np.deg2rad(self.phi_deg))
        self.R0, self.a = float(mesh.R0), float(mesh.a)
        self.nr, self.ntheta = int(mesh.nr) - 1, int(mesh.ntheta)
        self.dr, self.dtheta = float(mesh.dr), float(mesh.dtheta)
        if not np.allclose([self.nr*self.dr, self.ntheta*self.dtheta], [self.a, 2*np.pi], rtol=1e-12):
            raise ValueError('Plane bins require the standard uniform field mesh')
        self.rho_edges = np.arange(self.nr + 1) * self.dr
        # 0 and 2pi are the same native periodic theta node.
        self.theta_edges = np.arange(self.ntheta + 1) * self.dtheta
        if not np.isfinite(min_normal_speed) or min_normal_speed < 0:
            raise ValueError('Minimum normal speed must be finite and nonnegative')
        self.min_normal_speed = float(min_normal_speed)
        self.duration_s = None
        self.statistics = False
        self.counts = None

    def prepare_torch(self, device, statistics=False):
        self.counts = torch.zeros((2, self.nr, self.ntheta), dtype=torch.int64, device=device)
        self.statistics = statistics
        self.energy_sum = torch.zeros_like(self.counts, dtype=torch.float64)
        self.inverse_speed_sum = torch.zeros_like(self.energy_sum)
        self.density_excluded = torch.zeros_like(self.counts)

    def accumulate_torch(self, start, end, velocity=None, mass_kg=None):
        s0 = start[:, 0]*self.sin_phi + start[:, 1]*self.cos_phi
        s1 = end[:, 0]*self.sin_phi + end[:, 1]*self.cos_phi
        s0 = torch.where(s0.abs() <= self.plane_tolerance_m, 0., s0)
        s1 = torch.where(s1.abs() <= self.plane_tolerance_m, 0., s1)
        crossing = ((s0 < 0) & (s1 >= 0)) | ((s0 > 0) & (s1 <= 0))
        p0, p1 = start[crossing], end[crossing]
        a0, a1 = s0[crossing], s1[crossing]
        point = p0 + (a0 / (a0-a1))[:, None] * (p1-p0)
        major_r = point[:, 0]*self.cos_phi - point[:, 1]*self.sin_phi
        u, z = major_r-self.R0, point[:, 2]
        rho = torch.hypot(u, z)
        theta = torch.remainder(torch.atan2(z, u), 2*np.pi)
        valid = (major_r > 0) & (rho <= self.a)
        ir = torch.floor(rho[valid]/self.dr).long().clamp(0, self.nr-1)
        it = torch.floor(theta[valid]/self.dtheta).long().clamp(0, self.ntheta-1)
        # channel 0: increasing computational phi (decreasing signed distance).
        direction = (a1[valid] > a0[valid]).long()
        index = (direction*self.nr + ir)*self.ntheta + it
        self.counts.view(-1).scatter_add_(0, index, torch.ones_like(index))
        if self.statistics:
            v = velocity[crossing][valid]
            mass = mass_kg[crossing][valid]
            energy = .5*mass*torch.sum(v*v, dim=1)/1.602176634e-19
            speed = (v[:, 0]*self.sin_phi + v[:, 1]*self.cos_phi).abs()
            keep = speed > self.min_normal_speed
            self.energy_sum.view(-1).scatter_add_(0, index, energy)
            self.inverse_speed_sum.view(-1).scatter_add_(0, index[keep], 1./speed[keep])
            self.density_excluded.view(-1).scatter_add_(0, index[~keep], torch.ones_like(index[~keep]))

    def numpy_counts(self):
        if isinstance(self.counts, torch.Tensor):
            return self.counts.detach().cpu().numpy()
        return np.asarray(self.counts)

    def save(self, sim_io, *, dt, n_particles):
        from .plane_crossing_plots import plot_histogram
        directional = self.numpy_counts()
        counts = directional.sum(axis=0)
        area = .5*np.diff(self.rho_edges**2)[:, None]*np.diff(self.theta_edges)[None, :]
        path = Path(sim_io._outputPath(sim_io.data_dir, 'Plane_crossings.npz'))
        data = dict(schema_version=2, counts=counts, counts_increasing_phi=directional[0],
                    counts_decreasing_phi=directional[1], rho_edges_m=self.rho_edges,
                    theta_edges_rad=self.theta_edges, cell_area_m2=area,
                    phi_deg=self.phi_deg, R0_m=self.R0, vessel_radius_m=self.a,
                    dt_s=dt, n_particles=n_particles, plane_tolerance_m=self.plane_tolerance_m,
                    segment_convention='(start, end]; initial points excluded')
        if self.duration_s is not None:
            data['observation_duration_s'] = self.duration_s
        if self.statistics:
            for key, value in (('energy_sum_eV', self.energy_sum),
                               ('inverse_normal_speed_sum_s_m', self.inverse_speed_sum),
                               ('density_excluded_counts', self.density_excluded)):
                value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                data[key] = value.sum(axis=0)
                data[key+'_increasing_phi'] = value[0]
                data[key+'_decreasing_phi'] = value[1]
            data['min_normal_speed_m_s'] = self.min_normal_speed
            data['energy_method'] = 'Boris drift velocity before second collision half-step'
        np.savez_compressed(path, **data)
        sim_io.log.info('PLANE CROSSINGS: phi=%g deg, %d events; %s', self.phi_deg, counts.sum(), path)
        import matplotlib.pyplot as plt
        fig = plot_histogram(data, mode='count')
        sim_io.saveFig('Plane_crossings.png')
        plt.close(fig)
