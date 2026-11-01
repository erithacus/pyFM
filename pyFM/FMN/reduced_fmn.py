from ..mesh.reduced import ReducedMesh
from ..spectral.nearest_neighbor import knn_query
from .FMN import FMN


class ReducedFMN:
    """
    Reduced Functional Map Network for scalable multi-mesh matching.

    Architectural Pattern:
        This class uses **Composition / Delegation** rather than Inheritance.
        It encapsulates a vanilla :class:`~pyFM.FMN.FMN` instance (``self.fmn``)
        operating exclusively on the reduced Galerkin subspace
        (:class:`~pyFM.mesh.reduced.ReducedMesh`).

        During Consistent ZoomOut refinement, ``self.fmn`` executes in its own
        coordinate space where the "complete" mesh consists of the :math:`p` sample
        vertices with exact Galerkin mass matrix :math:`A_{\\text{red}}`. This avoids
        the dynamic dispatch pitfalls of inheritance, ensuring that internal ZoomOut
        iterations never accidentally trigger dense matrix uplifting or dimension mismatches.

        Dense uplifting via the partition-of-unity interpolation basis :attr:`~ReducedMesh.U`
        is exposed at the wrapper boundary via :meth:`get_LB` and :meth:`compute_p2p`
        when ``complete=True`` is requested by the user.

    Parameters
    ----------
    meshlist : list of TriMesh or ReducedMesh
        List of meshes in the network. If elements are dense :class:`~pyFM.mesh.trimesh.TriMesh`
        instances, they are automatically converted to :class:`ReducedMesh`.
    maps_dict : dict, optional
        Initial functional maps between pairs of meshes (i, j).
    n_samples : int, optional
        Target number of samples for ReducedMesh construction. Default is 1000.
    k : int, optional
        Number of reduced eigenvectors to compute. Default is 40.
    **kwargs : Any
        Additional parameters passed to :class:`~pyFM.mesh.reduced.ReducedMesh`.
    """

    def __init__(
        self,
        meshlist,
        maps_dict=None,
        n_samples=1000,
        k=40,
        **kwargs,
    ):
        self.dense_meshes = meshlist
        self.reduced_meshes = [
            m if isinstance(m, ReducedMesh) else ReducedMesh(m, n_samples=n_samples, k=k, **kwargs)
            for m in meshlist
        ]
        # Internal vanilla FMN operating strictly on reduced Galerkin coordinates
        self.fmn = FMN(self.reduced_meshes, maps_dict=maps_dict)
        self.p2p = {}

    def zoomout_refine(
        self,
        nit=10,
        step=1,
        subsample=None,
        **kwargs,
    ):
        """
        Refine functional maps using Consistent ZoomOut on the reduced network.

        Forces `subsample=None` on the internal FMN instance to ensure that all
        iterations remain strictly in the exact Galerkin reduced space preserving
        the reduced Riemannian mass metric :math:`A_{\\text{red}}`.

        Parameters
        ----------
        nit : int, optional
            Number of zoomout iterations (default: 10).
        step : int, optional
            Dimension increase at each iteration (default: 1).
        subsample : None, optional
            Ignored; forced to None to preserve the exact Galerkin inner product.
        **kwargs : Any
            Additional parameters passed to :meth:`pyFM.FMN.FMN.zoomout_refine`.
        """
        return self.fmn.zoomout_refine(nit=nit, step=step, subsample=None, **kwargs)

    def compute_CCLB(self, m=None, **kwargs):
        """
        Compute the Canonical Consistent Latent Basis on the reduced network.

        Parameters
        ----------
        m : int, optional
            Size of CCLB. If None, defaults to `int(0.9 * self.M)`.
        **kwargs : Any
            Additional parameters passed to :meth:`pyFM.FMN.FMN.compute_CCLB`.
        """
        if m is None:
            m = int(0.9 * self.fmn.M)
        return self.fmn.compute_CCLB(m, **kwargs)

    def get_LB(self, i, complete=True):
        """
        Return the latent basis (LB) for mesh i.

        Parameters
        ----------
        i : int
            Index of the mesh on which to return the latent basis.
        complete : bool, optional
            If True (default), uplifts the latent basis via the partition-of-unity
            basis :attr:`~ReducedMesh.U` to all :math:`N_i` continuous surface
            vertices: shape ``(N_i, m)``.
            If False, returns the latent basis on the :math:`p_i` reduced sample
            vertices: shape ``(p_i, m)``.

        Returns
        -------
        latent_basis : (N_i, m) or (p_i, m) np.ndarray
            Latent basis on mesh i.
        """
        if not complete:
            # Returns the reduced basis from internal FMN: (p_i, m)
            return self.fmn.get_LB(i, complete=False)

        # Uplift to dense continuous surface via partition of unity U: (N_i, m)
        cclb = self.fmn.CCLB[i]
        return self.reduced_meshes[i].get_dense_basis()[:, : self.fmn.M] @ cclb

    def compute_p2p(
        self,
        complete=True,
        n_jobs=None,
    ):
        """
        Compute vertex-to-vertex maps for each directed edge from the CCLB.

        Parameters
        ----------
        complete : bool, optional
            If True (default), lifts latent bases via :meth:`get_LB(i, complete=True)`
            to return full dense point-to-point maps :math:`(N_j,)` on continuous surface vertices.
            If False, returns subsampled point-to-point maps :math:`(p_j,)`.
        n_jobs : int, optional
            Number of parallel jobs for nearest-neighbor search.

        Returns
        -------
        p2p : dict of (int, int) -> np.ndarray
            Dictionary of pointwise maps from target meshes to source meshes.
        """
        if not complete:
            self.fmn.compute_p2p(complete=False, n_jobs=n_jobs)
            self.p2p = self.fmn.p2p
            return self.p2p

        # Dense uplifting: query nearest neighbors using dense latent bases
        dense_LB = [self.get_LB(i, complete=True) for i in range(self.fmn.n_meshes)]
        self.p2p = {
            (i, j): knn_query(dense_LB[i], dense_LB[j], k=1, n_jobs=n_jobs) for i, j in self.edges
        }
        self.fmn.p2p = self.p2p
        return self.p2p

    @property
    def maps(self):
        return self.fmn.maps

    @property
    def edges(self):
        return self.fmn.edges

    @property
    def CCLB(self):
        return self.fmn.CCLB

    @property
    def M(self):
        return self.fmn.M

    def __getattr__(self, name):
        return getattr(self.fmn, name)
