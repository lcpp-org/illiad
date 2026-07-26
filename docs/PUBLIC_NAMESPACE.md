# Public Python Namespace

The preferred public import path is `illiad.*`, with shared utilities grouped
under `illiad.utilities.*`. The former `classes.*`, `utility.*`, and
`plot_funcs.*` packages have been removed.

## Meshes

```python
from illiad.mesh import Mesh, TorchMesh
```

- `Mesh` wraps the NumPy/scipy-oriented mesh implementation.
- `TorchMesh` wraps the torch-backed mesh implementation used by the Boris
  workflow.

## IO

```python
from illiad.io import IOHandler
```

## Flux Workflow

The flux implementations are available as classes:

```python
from illiad.flux import FluxCalculator, FluxInterpolator, FluxGradientor
```

## Coordinate Transforms

```python
from illiad.utilities.coordtrans import RTP_to_XYZ, XYZ_to_RTP
```

## Particles and Solvers

```python
from illiad.particle import Particle, FieldLine, Ion
from illiad.poincare import Poincare
from illiad.boris import Boris
```

## Particle Initialization

```python
from illiad.utilities.point_generators import (
    generateSeedShells,
    generate_MB_velocities,
    ionInitializer,
)
```

`ionInitializer` returns the ion list, combined initial velocity/position
array, and launch-normal array in one shared emitter-major particle order.
`generate_MB_velocities` uses Maxwellian speeds and cosine-weighted
hemispherical directions about those launch normals.

## Run Configs

```python
from illiad.utilities.run_config import load_inputs_json, merge_input_params, normalize_phi_gens
```

## Plotting Helpers

```python
from illiad import plotting
```

Legacy plotting and analysis scripts are retained under `misc_scripts/` and
are not installed as a runtime package.
