# Public Python Namespace

The preferred public import path is `illiad.*`, with shared utilities grouped
under `illiad.utilities.*`. Former `classes.*`, `utility.*`, and
`plot_funcs.*` packages have been removed.

Detailed stability and method contracts are defined in
[PUBLIC_API.md](PUBLIC_API.md). In version 1.0.0, the installed commands,
documented JSON keys, version identifier, and run-configuration utilities are
stable. Documented research-class object models remain provisional.

## Version

```python
from illiad import __version__
```

## IO and Meshes

```python
from illiad.io import IOHandler
from illiad.mesh import Mesh, TorchMesh
```

`Mesh` is the NumPy/SciPy-oriented structured mesh. `TorchMesh` is the
torch-backed mesh used by Boris and `SOLTracer`.

## Poincare and Flux Analyses

```python
from illiad.poincare import Poincare
from illiad.flux import FluxCalculator, FluxInterpolator, FluxGradientor
```

`FluxCalculator` uses measured rotational transform and low-order rational
matching for island-chain selection. `FluxGradientor` can consume a newly
interpolated profile or an existing regular scalar field.

## SOL Analysis

```python
from illiad.sol import (
    CrossingChunk,
    NpyPlaneCrossingSource,
    PlaneCrossingSource,
    SOLDensity,
    SOLPotential,
    SOLRegularizer,
    SOLTracer,
    build_torch_magnetic_field,
    load_lcfs_boundary,
    load_poincare_settings,
    minimum_boundary_distance,
    open_plane_crossing_source,
    resolve_device,
)
```

`SOLTracer` is the official open-field-line analysis class. It builds
LCFS-exterior seed grids, traces both directions with a `TorchMesh`, saves
compact toroidal-plane crossings, and produces optional contour plots.

`SOLRegularizer` consumes plane crossing chunks and writes the regular
float64 `(phi, theta, rho)` connection-length field used by later SOL
profile analyses. `PlaneCrossingSource` defines the chunk interface, and
`NpyPlaneCrossingSource` adapts current compact and legacy expanded outputs.

`SOLDensity` and `SOLPotential` are the package-native profile stitchers.
They consume saved interior, connection-length, and Poincare artifacts and
share their LCFS geometry and attenuation implementation through
`illiad.sol.stitching`.

## Particles, Boris, and Collisions

```python
from illiad.particle import Particle, FieldLine, Ion
from illiad.boris import Boris
from illiad.collisions import (
    Collisions,
    kg_per_amu,
    kboltz,
    eps0,
    sqrt_pi,
    Li_mass,
    He_mass,
)
```

Ion-neutral collision selectors are `viscous_drag` and `langevin`. Ion-ion
collision selectors are `linear_fp` and `fokker_planck`. Use `None` to disable
either collision category programmatically.

## Particle Initialization

```python
from illiad.utilities.point_generators import (
    generateSeedShells,
    generate_MB_velocities,
    ionInitializer,
)
```

`ionInitializer` returns ions, combined initial velocity/position data, and
launch normals in one emitter-major order. `generate_MB_velocities` uses
Maxwellian speeds and cosine-weighted hemispherical directions.

## Coordinate Transforms

```python
from illiad.utilities.coordtrans import (
    RTP_to_XYZ,
    XYZ_to_RTP,
    XYZ_to_RTP2,
    RTP_to_XYZ_many,
    XYZ_to_RTP_many,
    rot_vecXYZ_byPHI,
    RTP_XYZ_JAC,
    RTP_XYZ_JAC2,
    axisShift,
    align_z_to_vector,
)
```

## Run Configuration

```python
from illiad.utilities.run_config import (
    load_inputs_json,
    merge_input_params,
    normalize_phi_gens,
)
```

## Plotting

```python
from illiad import plotting
```

Python imports from `misc_scripts` and `fastplotlib_tests` remain outside the
public namespace.
