# Public Python Namespace

The preferred public import path is `illiad.*`. The older `classes.*`,
`utility.*`, and `plot_funcs.*` imports remain available for current scripts,
but new user-facing examples should use the `illiad` namespace.

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

Use the snake-case aliases for public code:

```python
from illiad.flux import calculate_flux, interpolate_flux, build_electric_field
```

Compatibility aliases remain available:

```python
from illiad.flux import fluxCalculator, fluxInterpolator, fluxGradientor
```

The compatibility aliases call the same implementation and are kept to avoid
breaking current research scripts while the public API settles.

## Coordinate Transforms

```python
from illiad.coordtrans import RTP_to_XYZ, XYZ_to_RTP
```

## Particles and Solvers

```python
from illiad.particle import Particle, FieldLine, Ion
from illiad.poincare import Poincare
from illiad.boris import Boris
```

## Run Configs

```python
from illiad.run_config import load_inputs_json, merge_input_params, normalize_phi_gens
```
