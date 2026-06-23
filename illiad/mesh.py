"""Public mesh namespace for ILLIAD."""

__all__ = ["Mesh", "TorchMesh"]


def __getattr__(name):
    if name == "Mesh":
        from classes.mesh import Mesh

        return Mesh
    if name == "TorchMesh":
        from classes.meshNew import Mesh as TorchMesh

        return TorchMesh
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
