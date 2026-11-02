"""
Scalable Functional Maps on Dense Meshes
===============================================

When matching a collection of high-resolution meshes, standard Functional
Map Networks (FMN) can become computationally slow even when using subsample during ZoomOut refinement,
because computing dense Laplace-Beltrami eigendecompositions on each shape at initialization remains slow.

The `ReducedFMN` and `ReducedMesh` classes follow the scheme described in "Scalable and Efficient Functional Map Computations
on Dense Meshes". These classes wrap the equivalent `FMN` and `TriMesh` classes in a quasi-transparent way in order to keep the established workflow the same.

This gallery page demonstrates :class:`~pyFM.FMN.ReducedFMN`, a wrapper around
:class:`~pyFM.FMN.FMN` that automates mesh reduction, does Consistent ZoomOut and solves CCLB in the reduced subspace,
and encapsulates the one-shot uplifting to full dense point-to-point correspondences at the end.

First, we demonstrate pairwise ZoomOut refinement directly on :class:`~pyFM.mesh.ReducedMesh`
using a shape pair where ground truth is known (cat and lion,
see :ref:`sphx_glr_auto_examples_plot_05_maps_without_correspondence.py`).

Then, a benchmark of :class:`~pyFM.FMN.ReducedFMN` directly against standard :class:`~pyFM.FMN.FMN`,
once with subsample = None and once with subsample = 1000 on frames of the galloping camel dataset is provided.
Note that the camel dataset is not really dense, so the timings are for demonstrative purposes only.

"""

# sphinx_gallery_thumbnail_number = 2

import os
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from pyFM.FMN import FMN, ReducedFMN
from pyFM.mesh import ReducedMesh, TriMesh
from pyFM.refine.zoomout import mesh_zoomout_refine_p2p
from pyFM.spectral import knn_query, mesh_p2p_to_FM
from pyFM.viz import plot_mesh, vertices_to_rgb


def example_data(name):
    """Return the path to a mesh in the ``examples/data`` folder."""
    for folder in [Path.cwd(), *Path.cwd().parents]:
        candidate = folder / "examples" / "data" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{name!r} not found. These examples read data from the pyFM repository; "
        "clone it and run them there."
    )


# %%
# Pairwise ZoomOut on ReducedMesh with Ground Truth
# -------------------------------------------------
#
# We test pairwise ZoomOut refinement on :class:`~pyFM.mesh.ReducedMesh` instances where ground-truth
# correspondence is known (cat and lion), at least approximatively.
#
# The two shapes share an aligned pose, allowing initialization by nearest neighbours
# in :math:`\mathbb{R}^3`. We load both meshes as :class:`~pyFM.mesh.ReducedMesh`,
# initialize the map with :func:`~pyFM.spectral.knn_query`, and refine it using
# :func:`~pyFM.refine.zoomout.mesh_zoomout_refine_p2p` with ``k_init=10``, ``nit=16``,
# and ``step=5`` (reaching :math:`k = 10 + 16 \times 5 = 90`).
#
# Finally, we demonstrate :meth:`~pyFM.mesh.ReducedMesh.lift_fm_to_dense_p2p` to uplift the
# refined functional map from the reduced subspace to full dense point-to-point correspondence.


np.random.seed(137)


def error_function(mesh1, mesh2, gt_p2p_21, n_samples=300):
    """Return a function scoring a map from shape 2 to shape 1, in percent of sqrt_area."""
    m1 = getattr(mesh1, "parent", mesh1)
    m2 = getattr(mesh2, "parent", mesh2)
    sub = m2.farthest_point_sampling(n_samples)
    distances = m1.geodesic_from(gt_p2p_21[sub])  # (n1, n_samples)
    columns = np.arange(n_samples)

    def error(p2p_21):
        return float(1e2 * distances[p2p_21[sub], columns].mean() / m1.sqrt_area)

    return error


def show_map(mesh1, mesh2, p2p_21, title="", cpos=None):
    """Color shape 1 by position, and shape 2 by the colors pulled through the map."""
    m1 = getattr(mesh1, "parent", mesh1)
    m2 = getattr(mesh2, "parent", mesh2)
    colors1 = vertices_to_rgb(m1.vertices)

    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0, 0)
    plot_mesh(m1, scalars=colors1, pl=pl)
    pl.add_title("source", font_size=9)
    pl.subplot(0, 1)
    plot_mesh(m2, scalars=colors1[p2p_21], pl=pl)
    pl.add_title(title, font_size=9)
    pl.link_views()
    pl.show(cpos=cpos if cpos is not None else "zy")


cat = ReducedMesh.load(example_data("cat-00.off"), area_normalize=True, center=True, k=100)
lion = ReducedMesh.load(example_data("lion-00.off"), area_normalize=True, center=True, k=100)
gt_p2p_21 = np.loadtxt(example_data("lion2cat"), dtype=int)

print(
    f"Source (Cat) : {cat.parent.n_vertices} dense vertices -> {cat.n_vertices} reduced samples (k={cat.eigenvectors.shape[1]})"
)
print(
    f"Target (Lion): {lion.parent.n_vertices} dense vertices -> {lion.n_vertices} reduced samples (k={lion.eigenvectors.shape[1]})"
)

error = error_function(cat, lion, gt_p2p_21)

p2p_21_nn = knn_query(cat.vertices, lion.vertices, k=1)
FM_zo = mesh_zoomout_refine_p2p(
    p2p_21_nn, cat, lion, k_init=10, nit=16, step=5, return_p2p=False, verbose=False
)

print(
    f"Refined functional map    : shape {FM_zo.shape} (k_init=10 -> k_final={FM_zo.shape[0]})"
)

# Uplift the refined functional map to full dense point-to-point correspondence via ReducedMesh.lift_fm_to_dense_p2p
p2p_dense_zo = cat.lift_fm_to_dense_p2p(FM_zo, lion)
print(
    f"Dense uplifted p2p map    : shape {p2p_dense_zo.shape} (maps {lion.parent.n_vertices} lion vertices to cat) | error = {error(p2p_dense_zo):.2f}%"
)

show_map(cat, lion, p2p_dense_zo, "nearest neighbours + ZoomOut (dense uplift)", cpos="zy")

# %%
# Benchmark Dataset: Galloping Camel Collection
# ---------------------------------------------
#
# Having verified pairwise ZoomOut on ReducedMesh, we evaluate the scalability of
# :class:`~pyFM.FMN.ReducedFMN` against standard :class:`~pyFM.FMN.FMN` across a
# multi-shape network on the galloping camel dataset.

camel_dir = example_data("camel_gallop")

print("Loading raw camel meshes...")
raw_meshes = [
    TriMesh.load(camel_dir / f"camel-gallop-{i:02d}.off", area_normalize=True, center=True)
    for i in range(1, 11)
]

n_dense_vertices = raw_meshes[0].n_vertices
print(f"Loaded {len(raw_meshes)} meshes ({n_dense_vertices} vertices each).")

# Establish network edges and load precomputed pairwise maps from dataset files.
# Unlike simple Euclidean nearest neighbours which confuse legs during wide gallop strides,
# these initial maps preserve bilateral symmetry and topological limb correspondences.
raw_p2p_maps = {}
network_edges = []
for map_filename in sorted(os.listdir(camel_dir / "maps")):
    ind2, ind1 = map_filename.split("_to_")
    i, j = int(ind1) - 1, int(ind2) - 1
    network_edges.append((i, j))
    raw_p2p_maps[(i, j)] = np.loadtxt(camel_dir / "maps" / map_filename, dtype=int)


def mean_deviation_from_dense(p2p_0_dict, dense_reference):
    """Compute mean Euclidean vertex deviation relative to the unconstrained dense FMN baseline."""
    errors = []
    v0 = raw_meshes[0].vertices
    for j in range(1, len(raw_meshes)):
        diff = np.linalg.norm(v0[p2p_0_dict[j]] - v0[dense_reference[j]], axis=1).mean()
        errors.append(diff)
    return float(np.mean(errors))


def mean_roundtrip_error(p2p_to_0, p2p_from_0):
    """Compute mean round-trip bijectivity error (0 -> j -> 0) in Euclidean space."""
    errors = []
    v0 = raw_meshes[0].vertices
    for j in range(1, len(raw_meshes)):
        back = p2p_to_0[j][p2p_from_0[j]]
        diff = np.linalg.norm(v0[back] - v0, axis=1).mean()
        errors.append(diff)
    return float(np.mean(errors))


# %%
# Pipeline & Benchmark Configuration
# ----------------------------------
#
# Unified configuration parameters to ensure all benchmarks remain in sync:

benchmark_k = 150  # Target spectrum / basis size computed during preprocessing
benchmark_k_init = 20  # Dimension of initial functional maps before ZoomOut refinement
benchmark_nit = 15  # Number of Consistent ZoomOut iterations
benchmark_steps = 5  # Dimension increment per ZoomOut iteration (M_final = 20 + 15*5 = 95)
benchmark_samples = 1000  # Number of samples (FPS in Benchmark 2, p subspace in Benchmark 3)


# %%
# Benchmark 1: Standard Dense FMN (subsample=None)
# ------------------------------------------------
#
# Standard FMN uses the full set of mesh vertices during Consistent ZoomOut.
# We track: 1) LBO preprocessing, 2) FMN network solve, and 3) downstream retrieval of maps to frame 0.

print("\n--- Running Standard Dense FMN (subsample=None) ---")
# 1. Preprocessing: Dense LBO & Eigendecomposition + Initial Maps
t0 = time.perf_counter()
meshlist_dense = [m.copy().process(k=benchmark_k, intrinsic=True) for m in raw_meshes]

initial_maps_dense = {}
for i, j in network_edges:
    initial_maps_dense[(i, j)] = mesh_p2p_to_FM(
        raw_p2p_maps[(i, j)], meshlist_dense[i], meshlist_dense[j], dims=benchmark_k_init
    )
time_prep_dense = time.perf_counter() - t0

# 2. Network Optimization (ZoomOut + CCLB, without edge p2p computation)
t0 = time.perf_counter()
fmn_dense = FMN(meshlist_dense, initial_maps_dense)
fmn_dense.zoomout_refine(nit=benchmark_nit, step=benchmark_steps, subsample=None, verbose=False)
fmn_dense.compute_CCLB(m=int(0.9 * fmn_dense.M), verbose=False)
time_solve_dense = time.perf_counter() - t0

# 3. Downstream Retrieval: Retrieve full p2p maps of frames 1..9 to frame 0
t0 = time.perf_counter()
LB_dense = [fmn_dense.get_LB(i) for i in range(len(raw_meshes))]
p2p_0_dense = {j: knn_query(LB_dense[0], LB_dense[j], k=1) for j in range(1, len(raw_meshes))}
p2p_from_0_dense = {j: knn_query(LB_dense[j], LB_dense[0], k=1) for j in range(1, len(raw_meshes))}
time_retrieve_dense = time.perf_counter() - t0

time_total_dense = time_prep_dense + time_solve_dense + time_retrieve_dense
rt_dense = mean_roundtrip_error(p2p_0_dense, p2p_from_0_dense)
dev_dense = 0.0

print(
    f"Dense FMN: Prep: {time_prep_dense:.2f}s | "
    f"Solve FMN: {time_solve_dense:.2f}s | Retrieve 1..9->0: {time_retrieve_dense:.2f}s | "
    f"Total: {time_total_dense:.2f}s"
)
print(f"Round-Trip Error: {rt_dense:.4f} | Deviation from Dense: 0.0000 (ref)")

# %%
# Benchmark 2: Subsampled Dense FMN (subsample=benchmark_samples)
# ---------------------------------------------------------------
#
# Subsampled FMN reduces pairwise matching time during ZoomOut via a subsampled list of vertices chosen by FPS
# but still requires precomputing the full dense Laplace-Beltrami eigenspectrum.

print(f"\n--- Running Subsampled Dense FMN (subsample={benchmark_samples}) ---")
# 1. Preprocessing: Dense LBO & Eigendecomposition + Initial Maps
t0 = time.perf_counter()
meshlist_sub = [m.copy().process(k=benchmark_k, intrinsic=True) for m in raw_meshes]

initial_maps_sub = {}
for i, j in network_edges:
    initial_maps_sub[(i, j)] = mesh_p2p_to_FM(
        raw_p2p_maps[(i, j)], meshlist_sub[i], meshlist_sub[j], dims=benchmark_k_init
    )
time_prep_sub = time.perf_counter() - t0

# 2. Network Optimization (Subsampled ZoomOut + CCLB, without edge p2p computation)
t0 = time.perf_counter()
fmn_sub = FMN(meshlist_sub, initial_maps_sub)
fmn_sub.zoomout_refine(
    nit=benchmark_nit, step=benchmark_steps, subsample=benchmark_samples, verbose=False
)
fmn_sub.compute_CCLB(m=int(0.9 * fmn_sub.M), verbose=False)
time_solve_sub = time.perf_counter() - t0

# 3. Downstream Retrieval: Retrieve full p2p maps of frames 1..9 to frame 0
t0 = time.perf_counter()
LB_sub = [fmn_sub.get_LB(i) for i in range(len(raw_meshes))]
p2p_0_sub = {j: knn_query(LB_sub[0], LB_sub[j], k=1) for j in range(1, len(raw_meshes))}
p2p_from_0_sub = {j: knn_query(LB_sub[j], LB_sub[0], k=1) for j in range(1, len(raw_meshes))}
time_retrieve_sub = time.perf_counter() - t0

time_total_sub = time_prep_sub + time_solve_sub + time_retrieve_sub
rt_sub = mean_roundtrip_error(p2p_0_sub, p2p_from_0_sub)
dev_sub = mean_deviation_from_dense(p2p_0_sub, p2p_0_dense)

print(
    f"Subsampled FMN: Prep: {time_prep_sub:.2f}s | "
    f"Solve FMN: {time_solve_sub:.2f}s | Retrieve 1..9->0: {time_retrieve_sub:.2f}s | "
    f"Total: {time_total_sub:.2f}s"
)
print(f"Round-Trip Error: {rt_sub:.4f} | Deviation from Dense: {dev_sub:.4f}")


# %%
# Benchmark 3: ReducedFMN (Scalable Functional Map Network)
# ---------------------------------------------------------
#
# ReducedFMN projects each raw mesh onto a reduced subspace (p=benchmark_samples)
# and solves only a small (p x p) generalized eigensystem.
#
# Initial functional maps are computed directly on the sample coordinates via nearest-neighbor
# search in the reduced basis, avoiding any dense spectral operations during network initialization.

print(f"\n--- Running ReducedFMN (p={benchmark_samples}) ---")
# 1. Preprocessing: Galerkin Subspace & Reduced Eigendecomposition from raw meshes
t0 = time.perf_counter()
red_meshes = [
    ReducedMesh(m.copy(), n_samples=benchmark_samples, k=benchmark_k, verbose=False)
    for m in raw_meshes
]
time_prep_red = time.perf_counter() - t0

# Quantize precomputed disk maps to sample coordinates on each target mesh.
# This adaptation step is not counted in the reduced basis solve timing.
dense_to_sample = [knn_query(rm.vertices, rm.parent.vertices, k=1) for rm in red_meshes]

t_fm_red = time.perf_counter()
initial_maps_red = {}
for i, j in network_edges:
    p2p_red_ji = dense_to_sample[i][raw_p2p_maps[(i, j)][red_meshes[j].samples]]
    initial_maps_red[(i, j)] = mesh_p2p_to_FM(
        p2p_red_ji,
        red_meshes[i],
        red_meshes[j],
        dims=benchmark_k_init,
    )
time_prep_red += time.perf_counter() - t_fm_red

# 2. Network Optimization in Reduced Subspace (ZoomOut + CCLB, without edge p2p computation)
t0 = time.perf_counter()
fmn_red = ReducedFMN(red_meshes, maps_dict=initial_maps_red)
fmn_red.zoomout_refine(nit=benchmark_nit, step=benchmark_steps, verbose=False)
fmn_red.compute_CCLB(verbose=False)
time_solve_red = time.perf_counter() - t0

# 3. Downstream Retrieval: Uplift CCLB to dense continuous surface & match frames 1..9 to frame 0
t0 = time.perf_counter()
LB_red = [fmn_red.get_LB(i, complete=True) for i in range(len(raw_meshes))]
p2p_0_red = {j: knn_query(LB_red[0], LB_red[j], k=1) for j in range(1, len(raw_meshes))}
p2p_from_0_red = {j: knn_query(LB_red[j], LB_red[0], k=1) for j in range(1, len(raw_meshes))}
time_retrieve_red = time.perf_counter() - t0

time_total_red = time_prep_red + time_solve_red + time_retrieve_red
rt_red = mean_roundtrip_error(p2p_0_red, p2p_from_0_red)
dev_red = mean_deviation_from_dense(p2p_0_red, p2p_0_dense)

print(
    f"ReducedFMN: Prep: {time_prep_red:.2f}s | "
    f"Solve FMN: {time_solve_red:.2f}s | Retrieve (Uplift+KNN): {time_retrieve_red:.2f}s | "
    f"Total: {time_total_red:.2f}s"
)
print(f"Round-Trip Error: {rt_red:.4f} | Deviation from Dense: {dev_red:.4f}")

# %%
# Runtime & Accuracy Summary
# --------------------------
#
# Comparing the results highlights the power of Galerkin subspace reduction across
# offline spectral preparation, network optimization, and dense correspondence retrieval:

print("\n" + "=" * 89)
print("SCALABLE FMN BENCHMARK SUMMARY".center(89))
print("=" * 89)
print(
    f"{'Metric':<24} | {'Dense (Full)':<18} | {f'Dense (Sub {benchmark_samples})':<18} | {f'ReducedFMN (p={benchmark_samples})':<20}"
)
print("-" * 89)
# print(f"{'Mesh Vertices (N)':<24} | {n_dense_vertices:<18} | {f'{n_dense_vertices} (fps {benchmark_samples})':<18} | {fmn_red.reduced_meshes[0].n_samples:<20}")
print(
    f"{'1. Basis Prep Time':<24} | {f'{time_prep_dense:.2f}s':<18} | {f'{time_prep_sub:.2f}s':<18} | {f'{time_prep_red:.2f}s':<20}"
)
print(
    f"{'2. Solve FMN (CZO+CCLB)':<24} | {f'{time_solve_dense:.2f}s':<18} | {f'{time_solve_sub:.2f}s':<18} | {f'{time_solve_red:.2f}s':<20}"
)
print(
    f"{'3. Retrieve 1..9->0 Time':<24} | {f'{time_retrieve_dense:.2f}s':<18} | {f'{time_retrieve_sub:.2f}s':<18} | {f'{time_retrieve_red:.2f}s':<20}"
)
print(
    f"{'Total Pipeline Time':<24} | {f'{time_total_dense:.2f}s':<18} | {f'{time_total_sub:.2f}s':<18} | {f'{time_total_red:.2f}s':<20}"
)
print(
    f"{'FMN Solve Speedup':<24} | {'1.00x (ref)':<18} | {f'{time_solve_dense / time_solve_sub:.2f}x':<18} | {f'{time_solve_dense / time_solve_red:.2f}x':<20}"
)
print(
    f"{'Total Speedup vs Full':<24} | {'1.00x (ref)':<18} | {f'{time_total_dense / time_total_sub:.2f}x':<18} | {f'{time_total_dense / time_total_red:.2f}x':<20}"
)
print(
    f"{'Round-Trip Error':<24} | {f'{rt_dense:.4f}':<18} | {f'{rt_sub:.4f}':<18} | {f'{rt_red:.4f}':<20}"
)
print(
    f"{'Deviation from Dense':<24} | {'0.0000 (ref)':<18} | {f'{dev_sub:.4f}':<18} | {f'{dev_red:.4f}':<20}"
)
print("=" * 89)

# %%
# Qualitative Comparison: Point-to-Point Pullback
# ------------------------------------------------
#
# To qualitatively assess the correspondence across the three approaches, we pull back
# coordinate colors from Frame 0 onto Frame 5 using the retrieved point-to-point maps.
#
# Under a high-quality map, matching colors indicate matching anatomical points (head, legs,
# and tail align across poses). All three pipelines accurately preserve anatomical correspondence,
# while ReducedFMN achieves this at a fraction of the computational runtime.

colors0 = vertices_to_rgb(raw_meshes[0].vertices)
target_frame = 5

pl = pv.Plotter(shape=(1, 4))

pl.subplot(0, 0)
plot_mesh(raw_meshes[0], scalars=colors0, pl=pl)
pl.add_title("Frame 0 (Source)", font_size=9)

for col, (name, p2p_dict) in enumerate(
    [
        ("Dense (Full)", p2p_0_dense),
        (f"Dense (Sub {benchmark_samples})", p2p_0_sub),
        (f"ReducedFMN (p={benchmark_samples})", p2p_0_red),
    ],
    start=1,
):
    pl.subplot(0, col)
    plot_mesh(raw_meshes[target_frame], scalars=colors0[p2p_dict[target_frame]], pl=pl)
    pl.add_title(name, font_size=9)

pl.link_views()
pl.show(cpos="zy")
