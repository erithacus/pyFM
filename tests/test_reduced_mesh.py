"""Characterization tests for :class:`pyFM.mesh.ReducedMesh`.

These pin the observable behaviour of the class so it can be refactored safely.
Values are either derived from the reduced mesh itself or taken from the
underlying implementation on ``examples/data/cat-00.off``.
"""

import copy
import pickle
import warnings

import numpy as np
import pytest

from pyFM.FMN import ReducedFMN
from pyFM.functional import FunctionalMapping
from pyFM.mesh import ReducedMesh, TriMesh

# Reference quantities for cat-00.off
N_VERTICES_DENSE = 7207
N_FACES_DENSE = 14410
AREA = 0.35022939765342315

N_SAMPLES = 150
K_MODES = 15


@pytest.fixture
def cat_dense(cat_path):
    """Normalized, centered cat mesh with precomputed spectrum."""
    mesh = TriMesh.load(cat_path, area_normalize=True, center=True)
    mesh.process(k=25)
    return mesh


@pytest.fixture
def cat_red(cat_dense):
    """Standard ReducedMesh instance (150 samples, 15 eigenvectors)."""
    return ReducedMesh(cat_dense, n_samples=N_SAMPLES, k=K_MODES)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_build_from_trimesh(cat_red, cat_dense):
    assert isinstance(cat_red, TriMesh)
    assert isinstance(cat_red, ReducedMesh)
    assert cat_red.n_vertices == N_SAMPLES
    assert cat_red.n_samples == N_SAMPLES
    assert cat_red.vertices.shape == (N_SAMPLES, 3)
    assert cat_red.vertices.dtype == float


def test_build_from_arrays(cat_dense):
    rm = ReducedMesh(cat_dense.vertices, cat_dense.faces, n_samples=N_SAMPLES, k=10)
    assert isinstance(rm, ReducedMesh)
    assert rm.n_vertices == N_SAMPLES
    assert rm.parent.n_vertices == N_VERTICES_DENSE


def test_load_from_path(cat_path):
    rm = ReducedMesh.load(cat_path, n_samples=N_SAMPLES, k=10)
    assert isinstance(rm, ReducedMesh)
    assert rm.n_vertices == N_SAMPLES
    assert rm.parent.n_vertices == N_VERTICES_DENSE
    assert rm.name == "cat-00"


def test_deprecated_path_constructor(cat_path):
    with pytest.warns(DeprecationWarning, match="use ReducedMesh.load"):
        rm = ReducedMesh(cat_path, n_samples=N_SAMPLES, k=10)
    assert isinstance(rm, ReducedMesh)
    assert rm.n_vertices == N_SAMPLES


def test_parent_and_name_recorded(cat_red, cat_dense):
    assert cat_red.parent is cat_dense
    assert cat_red.dense_mesh is cat_dense
    assert cat_red.name == cat_dense.name


def test_point_cloud_structure(cat_red):
    """ReducedMesh operates as an immutable point-cloud TriMesh."""
    assert cat_red.is_point_cloud is True
    assert cat_red.faces is None
    assert cat_red.n_faces == 0


def test_rejects_malformed_inputs():
    with pytest.raises(TypeError, match="Expected TriMesh"):
        ReducedMesh(12345)
    with pytest.raises(TypeError, match="Expected TriMesh"):
        ReducedMesh([1, 2, 3])


# ---------------------------------------------------------------------------
# Constructor options
# ---------------------------------------------------------------------------


def test_process_params_dict(cat_dense):
    params = {"n_samples": 80, "k": 7}
    rm = ReducedMesh(cat_dense, process_params=params)
    assert rm.n_samples == 80
    assert rm.eigenvalues.shape == (7,)


def test_n_samples_option(cat_dense):
    rm = ReducedMesh(cat_dense, n_samples=80, k=5)
    assert rm.n_samples == 80
    assert rm.n_vertices == 80
    assert len(rm.samples) == 80


def test_k_option(cat_dense):
    rm = ReducedMesh(cat_dense, n_samples=100, k=12)
    assert rm.eigenvalues.shape == (12,)
    assert rm.eigenvectors.shape == (100, 12)


def test_verbose_option(cat_dense, capsys):
    _ = ReducedMesh(cat_dense, n_samples=50, k=5, verbose=True)
    captured = capsys.readouterr()
    assert "Constructing ReducedMesh" in captured.out or "Processing" in captured.out


# ---------------------------------------------------------------------------
# Geometric quantities
# ---------------------------------------------------------------------------


def test_area(cat_red, cat_dense):
    """The reduced mass matrix preserves total continuous surface area."""
    assert cat_red.area == pytest.approx(cat_dense.area, rel=1e-6)
    assert cat_red.sqrt_area == pytest.approx(np.sqrt(cat_dense.area), rel=1e-6)


def test_areas_are_consistent(cat_red, cat_dense):
    """Vertex areas and total area must agree."""
    assert cat_red.vertex_areas.shape == (N_SAMPLES,)
    assert cat_red.vertex_areas.sum() == pytest.approx(cat_dense.area, rel=1e-6)
    assert (cat_red.vertex_areas > 0).all()


def test_mass_matrix_area_conservation(cat_red, cat_dense):
    """1^T A_red 1 must equal total surface area of parent mesh."""
    ones = np.ones(cat_red.n_vertices)
    total_area = ones @ (cat_red.mass @ ones)
    assert total_area == pytest.approx(cat_dense.area, rel=1e-6)


# ---------------------------------------------------------------------------
# Point clouds
# ---------------------------------------------------------------------------


def test_is_point_cloud(cat_red):
    assert cat_red.is_point_cloud is True


@pytest.mark.parametrize(
    "operation",
    ["edges", "face_normals", "face_areas"],
)
def test_point_cloud_rejects_face_quantities(cat_red, operation):
    with pytest.raises(ValueError, match="no faces"):
        getattr(cat_red, operation)


def test_point_cloud_rejects_geodesics(cat_red):
    with pytest.raises(ValueError, match="no faces"):
        cat_red.geodesic_from(0)


def test_repr(cat_red):
    r = repr(cat_red)
    assert "ReducedMesh" in r
    assert str(N_SAMPLES) in r
    assert "point cloud" in r


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


def test_process_computes_reduced_spectrum(cat_red):
    assert cat_red.eigenvalues.shape == (K_MODES,)
    assert cat_red.eigenvectors.shape == (N_SAMPLES, K_MODES)
    assert cat_red.stiffness.shape == (N_SAMPLES, N_SAMPLES)
    assert cat_red.mass.shape == (N_SAMPLES, N_SAMPLES)


def test_process_returns_self(cat_red):
    assert cat_red.process(k=5) is cat_red


def test_compute_spectrum_returns_reduced_spectrum(cat_red):
    evals, evecs = cat_red.compute_spectrum(k=10, return_spectrum=True)
    assert evals.shape == (10,)
    assert evecs.shape == (N_SAMPLES, 10)
    assert cat_red.eigenvalues.shape == (10,)
    assert cat_red.eigenvectors.shape == (N_SAMPLES, 10)
    assert cat_red.evals_red.shape == (10,)
    assert cat_red.evecs_red.shape == (N_SAMPLES, 10)


def test_first_eigenvalue_is_zero(cat_red):
    assert abs(cat_red.eigenvalues[0]) < 1e-6
    assert (np.diff(cat_red.eigenvalues) >= -1e-10).all()


def test_eigenvalues_are_sorted(cat_red):
    diffs = np.diff(cat_red.eigenvalues)
    assert (diffs >= -1e-10).all()


def test_process_truncates_existing_spectrum(cat_red):
    evals_before = cat_red.eigenvalues.copy()
    cat_red.process(k=8)
    assert cat_red.eigenvalues.shape == (8,)
    assert cat_red.eigenvectors.shape == (N_SAMPLES, 8)
    np.testing.assert_allclose(cat_red.eigenvalues, evals_before[:8])


def test_eigenvectors_are_mass_orthonormal(cat_red):
    gram = cat_red.eigenvectors.T @ (cat_red.mass @ cat_red.eigenvectors)
    np.testing.assert_allclose(gram, np.eye(K_MODES), atol=1e-5)


def test_operators_are_symmetric(cat_red):
    mass = cat_red.mass.toarray() if hasattr(cat_red.mass, "toarray") else cat_red.mass
    stiff = (
        cat_red.stiffness.toarray() if hasattr(cat_red.stiffness, "toarray") else cat_red.stiffness
    )
    np.testing.assert_allclose(mass, mass.T, atol=1e-10)
    np.testing.assert_allclose(stiff, stiff.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Projection / inner products
# ---------------------------------------------------------------------------


def test_project_unproject_shapes(cat_red):
    f = cat_red.vertices[:, 0].copy()
    coeffs = cat_red.project(f)
    assert coeffs.shape == (K_MODES,)
    assert cat_red.unproject(coeffs).shape == (N_SAMPLES,)
    assert cat_red.project(f, k=5).shape == (5,)


def test_project_multiple_functions(cat_red):
    coeffs = cat_red.project(cat_red.vertices)
    assert coeffs.shape == (K_MODES, 3)
    assert cat_red.unproject(coeffs).shape == (N_SAMPLES, 3)


def test_project_rejects_too_many_modes(cat_red):
    with pytest.raises(ValueError):
        cat_red.project(cat_red.vertices[:, 0], k=500)


def test_l2_inner_matches_mass_matrix(cat_red):
    f = cat_red.vertices[:, 0].copy()
    g = cat_red.vertices[:, 1].copy()
    assert cat_red.l2_inner(f, g) == pytest.approx(f @ (cat_red.mass @ g))
    assert cat_red.l2_sqnorm(f) == pytest.approx(f @ (cat_red.mass @ f))


def test_h1_inner_matches_stiffness_matrix(cat_red):
    f = cat_red.vertices[:, 0].copy()
    g = cat_red.vertices[:, 1].copy()
    assert cat_red.h1_inner(f, g) == pytest.approx(f @ (cat_red.stiffness @ g))
    assert cat_red.h1_sqnorm(f) == pytest.approx(f @ (cat_red.stiffness @ f))


def test_integrate_constant_gives_area(cat_red, cat_dense):
    assert cat_red.integrate(np.ones(N_SAMPLES)) == pytest.approx(cat_dense.area, rel=1e-6)


# ---------------------------------------------------------------------------
# Subspace lifting & Partition of Unity
# ---------------------------------------------------------------------------


def test_partition_of_unity(cat_red):
    """Interpolation basis U rows must sum to 1 everywhere on the dense surface."""
    u_row_sums = np.asarray(cat_red.U.sum(axis=1)).squeeze()
    np.testing.assert_allclose(u_row_sums, 1.0, atol=1e-10)


def test_lift_signal_constant(cat_red):
    """Constant signal 1 on reduced mesh lifts to constant 1 on dense mesh."""
    signal_red = np.ones(cat_red.n_samples)
    signal_dense = cat_red.lift_signal(signal_red)
    assert signal_dense.shape == (N_VERTICES_DENSE,)
    np.testing.assert_allclose(signal_dense, 1.0, atol=1e-10)


def test_get_dense_basis_orthonormality(cat_red, cat_dense):
    """Lifted basis (U Phi_red) is orthonormal with respect to dense mass matrix A."""
    dense_basis = cat_red.get_dense_basis()
    assert dense_basis.shape == (N_VERTICES_DENSE, K_MODES)
    gram = dense_basis.T @ (cat_dense.A @ dense_basis)
    np.testing.assert_allclose(gram, np.eye(K_MODES), atol=1e-5)


def test_dense_basis_caching(cat_red):
    """get_dense_basis caches its result and invalidates upon process()."""
    b1 = cat_red.get_dense_basis()
    b2 = cat_red.get_dense_basis()
    assert b1 is b2
    cat_red.process(k=8)
    b3 = cat_red.get_dense_basis()
    assert b3.shape == (N_VERTICES_DENSE, 8)
    assert b3 is not b1


# ---------------------------------------------------------------------------
# Geodesics and sampling
# ---------------------------------------------------------------------------


def test_farthest_point_sampling_rejects(cat_red):
    """Calling farthest_point_sampling on ReducedMesh strictly raises ValueError."""
    with pytest.raises(ValueError, match="Cannot subsample a ReducedMesh"):
        cat_red.farthest_point_sampling(50)

    with pytest.raises(ValueError, match="Cannot subsample a ReducedMesh"):
        cat_red.farthest_point_sampling(200)

    with pytest.raises(ValueError, match="Cannot subsample a ReducedMesh"):
        cat_red.extract_fps(50)


# ---------------------------------------------------------------------------
# Copying and serialization — regressions for the deepcopy bug
# ---------------------------------------------------------------------------


def test_deepcopy_is_independent(cat_red):
    other = copy.deepcopy(cat_red)
    assert other is not cat_red
    assert other.n_vertices == cat_red.n_vertices
    np.testing.assert_allclose(other.eigenvalues, cat_red.eigenvalues)
    other.vertices[0] += 10.0
    assert not np.allclose(other.vertices[0], cat_red.vertices[0])


def test_deepcopy_shares_parent_reference(cat_red):
    """Crucial memory hygiene: deepcopy must share the parent mesh reference."""
    other = copy.deepcopy(cat_red)
    assert other.parent is cat_red.parent
    assert other.dense_mesh is cat_red.dense_mesh


def test_pickle_roundtrip(cat_red):
    pickled = pickle.dumps(cat_red)
    reloaded = pickle.loads(pickled)
    assert isinstance(reloaded, ReducedMesh)
    assert reloaded.n_vertices == cat_red.n_vertices
    np.testing.assert_allclose(reloaded.vertices, cat_red.vertices)
    np.testing.assert_allclose(reloaded.eigenvalues, cat_red.eigenvalues)


# ---------------------------------------------------------------------------
# Backwards compatibility and aliases
# ---------------------------------------------------------------------------


def test_short_operator_aliases_are_silent(cat_red):
    """W, A, W_red, A_red are kept as short forms; they must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert cat_red.W is cat_red.stiffness
        assert cat_red.A is cat_red.mass
        assert cat_red.W_red is cat_red.stiffness
        assert cat_red.A_red is cat_red.mass
        assert cat_red.evecs_red is cat_red.eigenvectors
        assert cat_red.evals_red is cat_red.eigenvalues
        assert cat_red.dense_mesh is cat_red.parent


def test_reduced_and_dense_operator_shapes(cat_red, cat_dense):
    """ReducedMesh.A and .W return reduced (p, p) matrices; dense operators are cached."""
    # Reduced (p, p) matrices
    assert cat_red.A.shape == (N_SAMPLES, N_SAMPLES)
    assert cat_red.W.shape == (N_SAMPLES, N_SAMPLES)
    assert cat_red.mass.shape == (N_SAMPLES, N_SAMPLES)
    assert cat_red.stiffness.shape == (N_SAMPLES, N_SAMPLES)
    assert cat_red.A_red.shape == (N_SAMPLES, N_SAMPLES)
    assert cat_red.W_red.shape == (N_SAMPLES, N_SAMPLES)

    # Dense (N, N) parent operators cached for lifting
    assert cat_red.A_dense.shape == (N_VERTICES_DENSE, N_VERTICES_DENSE)
    assert cat_red.W_dense.shape == (N_VERTICES_DENSE, N_VERTICES_DENSE)
    assert cat_red.mass_dense.shape == (N_VERTICES_DENSE, N_VERTICES_DENSE)
    assert cat_red.stiffness_dense.shape == (N_VERTICES_DENSE, N_VERTICES_DENSE)
    assert cat_red.parent.A.shape == (N_VERTICES_DENSE, N_VERTICES_DENSE)
    assert cat_red.parent.W.shape == (N_VERTICES_DENSE, N_VERTICES_DENSE)


def test_sub_aliases_are_absent(cat_red):
    """Obsolete _sub aliases must not be defined on ReducedMesh."""
    for attr in (
        "A_sub",
        "W_sub",
        "evals_sub",
        "evecs_sub",
        "subsample",
    ):
        assert not hasattr(cat_red, attr), f"Obsolete attribute {attr} should not exist"


# ---------------------------------------------------------------------------
# Functional maps integration
# ---------------------------------------------------------------------------


def test_pairwise_functional_mapping(cat_dense):
    """Verify end-to-end OO FunctionalMapping workflow with ReducedMesh."""
    rm1 = ReducedMesh(cat_dense, n_samples=200, k=25)
    rm2 = ReducedMesh(cat_dense, n_samples=200, k=25)

    model = FunctionalMapping(rm1, rm2)
    model.preprocess(descr_type="WKS", k_process=22, k_descr=15, verbose=False)
    model.fit(K=(15, 15), w_orient=0, verbose=False)

    assert model.FM_12.shape == (15, 15)

    FM_zo = model.zoomout_refine(nit=3, step=2)
    assert FM_zo.shape == (21, 21)

    p2p_sub = model.get_p2p(FM_zo)
    assert p2p_sub.shape == (200,)

    # Explicit dense lifting
    p2p_dense = rm1.lift_fm_to_dense_p2p(FM_zo, rm2)
    assert p2p_dense.shape == (cat_dense.n_vertices,)


def test_reduced_fmn_get_lb_dense_and_reduced(cat_dense):
    """Verify get_LB returns dense (N, m) when complete=True, reduced (p, m) when False."""
    dense_meshes = [
        TriMesh.load(cat_dense.path, area_normalize=True, center=True).process(k=25)
        for _ in range(3)
    ]
    maps = {(i, j): np.eye(15) for i in range(3) for j in range(3) if i != j}

    net = ReducedFMN(dense_meshes, maps_dict=maps, n_samples=180, k=25)
    net.zoomout_refine(nit=2, step=2, weight_type="adjacency")
    net.compute_CCLB()

    # Default complete=True lifts to dense: (N, m)
    lb_dense = net.get_LB(0, complete=True)
    assert lb_dense.shape == (N_VERTICES_DENSE, net.m_cclb)

    # complete=False returns reduced subspace basis: (p, m)
    lb_red = net.get_LB(0, complete=False)
    assert lb_red.shape == (180, net.m_cclb)


def test_reduced_fmn_network_workflow(cat_dense):
    """Verify end-to-end ReducedFMN workflow on a multi-mesh network."""
    dense_meshes = [
        TriMesh.load(cat_dense.path, area_normalize=True, center=True).process(k=25)
        for _ in range(3)
    ]
    maps = {(i, j): np.eye(15) for i in range(3) for j in range(3) if i != j}

    net = ReducedFMN(dense_meshes, maps_dict=maps, n_samples=180, k=25)
    net.zoomout_refine(nit=2, step=2, weight_type="adjacency")
    net.compute_CCLB()

    # Subsampled maps
    p2p_sub = net.compute_p2p(complete=False)
    assert len(p2p_sub) == 6
    for p in p2p_sub.values():
        assert p.shape == (180,)

    # Dense maps (encapsulated lifting)
    dense_p2p = net.compute_p2p(complete=True)
    assert len(dense_p2p) == 6
    for p in dense_p2p.values():
        assert p.shape == (cat_dense.n_vertices,)


# ---------------------------------------------------------------------------
# Sampling backend and graceful fallback tests
# ---------------------------------------------------------------------------


def test_sampling_backend_selection(cat_dense):
    """Test backend parameter handling in dense_mesh sampling."""
    from pyFM.mesh.dense_mesh.utils import poisson_sample_mesh

    # Force native FPS
    samples_fps = poisson_sample_mesh(cat_dense, n_samples=50, backend="fps")
    assert len(samples_fps) == 50
    assert len(np.unique(samples_fps)) == 50

    # Auto mode works regardless of pymeshlab presence
    samples_auto = poisson_sample_mesh(cat_dense, n_samples=50, backend="auto")
    assert len(samples_auto) == 50

    # Unknown backend raises ValueError
    with pytest.raises(ValueError, match="Unknown sampling backend"):
        poisson_sample_mesh(cat_dense, n_samples=50, backend="invalid_backend")


def test_sampling_backend_meshlab_missing(cat_dense):
    """backend='meshlab' must raise ImportError when pymeshlab is not installed."""
    from pyFM.mesh.dense_mesh import utils

    if utils.pymeshlab is None:
        with pytest.raises(ImportError, match="pymeshlab is required"):
            utils.poisson_sample_mesh(cat_dense, n_samples=50, backend="meshlab")

