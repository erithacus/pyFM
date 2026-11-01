from .trimesh import TriMesh


def __getattr__(name):
    if name == "ReducedMesh":
        from .reduced import ReducedMesh

        globals()["ReducedMesh"] = ReducedMesh
        return ReducedMesh
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TriMesh",
    "ReducedMesh",
]
