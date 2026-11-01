import copy
import warnings
from pathlib import Path

import numpy as np

from ..spectral import FM_to_p2p
from .trimesh import TriMesh


class ReducedMesh(TriMesh):
    """
    Subsampled spectral representation for a triangle mesh.

    Provides a wrapper for TriMesh that exposes only a reduced spectral subspace
    based on "Scalable and Efficient Functional Computations on Dense Meshes".

    It uses the code originally from `RobinMagnet/Scalable_FM <https://github.com/RobinMagnet/Scalable_FM>`_.

    The mesh is represented by a subset of `p` sample vertices with a reduced set of area and stiffness matrices.
    Its behaviour is transparent to `FMN` and `FunctionalMapping`.
    Not all methods of `TriMesh` are consistent in the reduced mode.
    However, the intended usage with `ReducedFMN` ensures that the reduced basis is uplifted as the final step.


    Parameters
    ----------
    mesh_or_path : TriMesh or str or Path
        A dense :class:`~pyFM.mesh.trimesh.TriMesh` instance or path to a 3D mesh file.
    n_samples : int, optional
        Target number of samples for the reduced basis. Default is 3000.
    k : int or None, optional
        Number of reduced eigenvectors to compute and store. Default is 100.
    verbose : bool, optional
        Whether to print progress during subspace construction. Default is False.
    **kwargs : Any
        Additional parameters passed to :func:`~pyFM.mesh.dense_mesh.process_mesh`
        (e.g. ``dist_ratio``, ``adapt_radius``, ``self_weight_limit``, ``backend``).

    Attributes
    ----------
    parent : TriMesh
        Underlying full-resolution mesh, also available as :attr:`dense_mesh`.
    U : scipy.sparse.csr_matrix
        (n, p) partition-of-unity interpolation matrix mapping functions from sample
        vertices to dense mesh vertices.
    samples : (p,) np.ndarray
        Indices of sample vertices on the dense mesh.
    n_samples : int
        Number of sample vertices `p`.
    mass : scipy.sparse.csr_matrix
        (p, p) reduced mass / area operator, also available as :attr:`A`.
    stiffness : scipy.sparse.csr_matrix
        (p, p) reduced stiffness / cotangent Laplacian operator, also available as :attr:`W`.
    eigenvalues : (k,) np.ndarray
        Eigenvalues of the reduced generalized eigenvalue problem.
    eigenvectors : (p, k) np.ndarray
        Eigenvectors of the reduced generalized eigenvalue problem.
    """

    @classmethod
    def load(
        cls,
        path,
        *,
        n_samples=1000,
        k=50,
        process_params=None,
        area_normalize=False,
        center=False,
        normalize=False,
        rotation=None,
        translation=None,
        name=None,
        verbose=False,
        **kwargs,
    ):
        """
        Read a mesh from a file and construct its reduced subspace representation.

        Parameters
        ----------
        path : str or os.PathLike
            Path to the mesh file to read.
        n_samples : int, optional
            Number of sample points for the reduced subspace (default: 1000).
        k : int, optional
            Number of reduced eigenvectors to compute (default: 50).
        process_params : dict, optional
            Optional dictionary of reduction parameters.
        area_normalize : bool, optional
            Whether to normalize the mesh surface area to 1.
        center : bool, optional
            Whether to center the mesh at the origin.
        normalize : bool, optional
            Shorthand for `area_normalize=True` and `center=True`.
        rotation : (3, 3) np.ndarray, optional
            Rigid rotation matrix applied to the mesh.
        translation : (3,) np.ndarray, optional
            Translation vector applied to the mesh.
        name : str, optional
            Name assigned to the mesh.
        verbose : bool, optional
            Whether to print construction logs.
        **kwargs : Any
            Additional reduction parameters passed to :func:`~pyFM.mesh.dense_mesh.process_mesh`.

        Returns
        -------
        mesh : ReducedMesh
            The constructed reduced mesh representation.
        """
        dense_mesh = TriMesh.load(
            path,
            area_normalize=area_normalize,
            center=center,
            normalize=normalize,
            rotation=rotation,
            translation=translation,
            name=name,
        )
        return cls(
            dense_mesh,
            n_samples=n_samples,
            k=k,
            process_params=process_params,
            verbose=verbose,
            **kwargs,
        )

    def __init__(
        self,
        mesh_or_vertices,
        faces=None,
        *,
        n_samples=1000,
        k=50,
        process_params=None,
        verbose=False,
        area_normalize=False,
        center=False,
        normalize=False,
        rotation=None,
        translation=None,
        name=None,
        **kwargs,
    ):
        if process_params is not None:
            n_samples = process_params.get("n_samples", n_samples)
            k = process_params.get("k", k)
            verbose = process_params.get("verbose", verbose)
            for key, val in process_params.items():
                if key not in ("n_samples", "k", "verbose"):
                    kwargs[key] = val

        if isinstance(mesh_or_vertices, (str, Path)):
            warnings.warn(
                "Building a ReducedMesh from a file path is deprecated, "
                "use ReducedMesh.load(path) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            dense_mesh = TriMesh.load(
                mesh_or_vertices,
                area_normalize=area_normalize,
                center=center,
                normalize=normalize,
                rotation=rotation,
                translation=translation,
                name=name,
            )
        elif isinstance(mesh_or_vertices, TriMesh):
            dense_mesh = mesh_or_vertices
        elif isinstance(mesh_or_vertices, np.ndarray):
            dense_mesh = TriMesh(
                mesh_or_vertices,
                faces=faces,
                area_normalize=area_normalize,
                center=center,
                normalize=normalize,
                rotation=rotation,
                translation=translation,
                name=name,
            )
        else:
            raise TypeError(
                f"Expected TriMesh, vertices array, or file path, got {type(mesh_or_vertices).__name__}"
            )

        # Ensure dense Laplacian operators are computed on parent
        if not hasattr(dense_mesh, "W") or dense_mesh.W is None or dense_mesh.A is None:
            dense_mesh.process(k=0)

        self.parent = dense_mesh
        self.dense_mesh = self.parent
        self.verbose = verbose
        self._dense_basis_cache = None

        if verbose:
            print(f"Constructing ReducedMesh with ~{n_samples} samples...")

        from .dense_mesh import process_mesh

        # Process dense mesh into reduced subspace operators
        U, A_bar, W_bar, subsample = process_mesh(
            dense_mesh,
            n_samples=n_samples,
            verbose=verbose,
            **kwargs,
        )

        self.U = U
        self.samples = subsample
        self.n_samples = len(subsample)

        # Initialize as a valid point-cloud TriMesh on sample coordinates
        sample_vertices = dense_mesh.vertices[subsample]
        super().__init__(vertices=sample_vertices, faces=None, name=dense_mesh.name)

        # Populate reduced operators
        self.mass = A_bar
        self.stiffness = W_bar
        self.A_red = self.mass
        self.W_red = self.stiffness

        # Compute reduced eigendecomposition
        k_target = k if k is not None else 100
        self.compute_spectrum(k=k_target, return_spectrum=False, verbose=verbose)

    def __deepcopy__(self, memo):
        """Deepcopy reduced arrays while keeping parent dense mesh as a shared reference."""
        other = type(self).__new__(type(self))
        memo[id(self)] = other
        for key, value in self.__dict__.items():
            if key in ("parent", "dense_mesh"):
                other.__dict__[key] = value
            elif key in getattr(self, "_SOLVER_ATTRIBUTES", ()):
                other.__dict__[key] = None
            else:
                other.__dict__[key] = copy.deepcopy(value, memo)
        return other

    def compute_spectrum(self, k=None, return_spectrum=True, verbose=False):
        """
        Compute the reduced Laplace-Beltrami spectrum on (stiffness, mass).

        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Defaults to current count or 100.
        return_spectrum : bool, optional
            Whether to return the computed spectrum.
        verbose : bool, optional
            Whether to print progress.

        Returns
        -------
        eigenvalues : (k,) np.ndarray, optional
            Computed reduced eigenvalues.
        eigenvectors : (p, k) np.ndarray, optional
            Computed reduced eigenvectors.
        """
        from .dense_mesh import get_approx_spectrum

        k_target = (
            k
            if k is not None
            else (
                self.eigenvalues.shape[0] if getattr(self, "eigenvalues", None) is not None else 100
            )
        )
        if verbose:
            print(
                f"Computing {k_target} reduced eigenvectors on ({self.n_samples}x{self.n_samples}) system..."
            )

        evals_red, evecs_red = get_approx_spectrum(
            self.stiffness, self.mass, k=k_target, verbose=verbose
        )

        self.eigenvalues = evals_red
        self.eigenvectors = evecs_red
        self._dense_basis_cache = None

        # Stabilize eigenvector signs against parent dense mesh if available
        if (
            getattr(self, "parent", None) is not None
            and getattr(self.parent, "eigenvectors", None) is not None
            and self.parent.eigenvectors.shape[1] > 0
        ):
            k_overlap = min(self.eigenvectors.shape[1], self.parent.eigenvectors.shape[1])
            dense_basis_k = self.U @ self.eigenvectors[:, :k_overlap]
            signs = np.sign(
                np.einsum(
                    "nk,nk->k",
                    dense_basis_k,
                    self.parent.A @ self.parent.eigenvectors[:, :k_overlap],
                )
            )
            signs[signs == 0] = 1.0
            self.eigenvectors[:, :k_overlap] *= signs[None, :]

        # Subspace aliases to track reduced state
        self.evals_red = self.eigenvalues
        self.evecs_red = self.eigenvectors

        if return_spectrum:
            return self.eigenvalues, self.eigenvectors

    def process(self, k=None, verbose=False, **kwargs):
        """
        Ensure the reduced mesh has at least `k` eigenvectors available.

        Compatible with the :meth:`TriMesh.process` signature called inside
        :meth:`~pyFM.functional.FunctionalMapping.preprocess`.

        Parameters
        ----------
        k : int, optional
            Target number of eigenvectors. If None or less than current count,
            slices existing eigenvectors.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        self : ReducedMesh
        """
        if k is not None:
            if self.eigenvalues is not None and k <= len(self.eigenvalues):
                self.eigenvectors = self.eigenvectors[:, :k]
                self.eigenvalues = self.eigenvalues[:k]
                self.evals_red = self.eigenvalues
                self.evecs_red = self.eigenvectors
                self._dense_basis_cache = None
            else:
                self.compute_spectrum(k=k, return_spectrum=False, verbose=verbose)
        return self

    def farthest_point_sampling(self, size, **kwargs):
        """
        Subsampling is strictly disallowed on a ReducedMesh.

        Raises
        ------
        ValueError
            Always raised because ReducedMesh is already an exact Galerkin
            reduced subspace. Subsampling it further violates its mathematical
            operators. Pass ``subsample=None`` in ``zoomout_refine``.
        """
        raise ValueError(
            f"Cannot subsample a ReducedMesh ({self.n_vertices} vertices). "
            "ReducedMesh is already an exact Galerkin reduced subspace; "
            "subsampling it further violates its mathematical metric. "
            "In zoomout_refine() or FunctionalMapping, pass subsample=None."
        )

    extract_fps = farthest_point_sampling

    # --- Dense Parent Operators (Cached for Lifting) ---

    @property
    def A_dense(self):
        """(N, N) dense mass / area matrix of the parent mesh."""
        return self.parent.A

    @property
    def W_dense(self):
        """(N, N) dense stiffness / Laplacian matrix of the parent mesh."""
        return self.parent.W

    @property
    def mass_dense(self):
        """(N, N) dense mass operator of the parent mesh."""
        return self.parent.mass

    @property
    def stiffness_dense(self):
        """(N, N) dense stiffness operator of the parent mesh."""
        return self.parent.stiffness

    # --- Lifting and Projection Helpers ---

    def get_dense_basis(self):
        """
        Compute the dense basis functions by lifting the reduced eigenvectors.

        Phi_dense = U @ Phi_reduced (N x k).

        Returns
        -------
        Phi_dense : (N, k) np.ndarray
            Lifted orthogonal basis on the full dense mesh.
        """
        if (
            self._dense_basis_cache is None
            or self._dense_basis_cache.shape[1] != self.eigenvectors.shape[1]
        ):
            if getattr(self, "verbose", False):
                print(f"Lifting basis to dense mesh ({self.parent.n_vertices} vertices)...")
            self._dense_basis_cache = self.U @ self.eigenvectors
        return self._dense_basis_cache

    def lift_signal(self, signal_red):
        """
        Lift a signal defined on sample vertices to the full dense mesh via U.

        Parameters
        ----------
        signal_red : (p,) or (p, d) np.ndarray
            Signal defined on the reduced sample vertices.

        Returns
        -------
        signal_dense : (N,) or (N, d) np.ndarray
            Interpolated signal on the dense mesh vertices.
        """
        return self.U @ signal_red

    def lift_fm_to_dense_p2p(
        self,
        FM,
        target_mesh,
        use_adj=True,
        n_jobs=None,
    ):
        """
        Compute a dense point-to-point map from source (self) to target_mesh.

        Parameters
        ----------
        FM : (k2, k1) np.ndarray
            Functional map mapping functions from source (self) to target.
        target_mesh : ReducedMesh or TriMesh
            Target mesh.
        use_adj : bool, optional
            Whether to use the adjoint embedding (default: True).
        n_jobs : int, optional
            Number of parallel jobs for nearest-neighbor search.

        Returns
        -------
        p2p_21 : (N_target,) np.ndarray
            Pointwise map from target vertices to source vertices.
        """
        dense_basis_source = self.get_dense_basis()
        if hasattr(target_mesh, "get_dense_basis"):
            dense_basis_target = target_mesh.get_dense_basis()
        else:
            dense_basis_target = target_mesh.eigenvectors

        return FM_to_p2p(FM, dense_basis_source, dense_basis_target, use_adj=use_adj, n_jobs=n_jobs)
