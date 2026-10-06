"""The two nearest-neighbour backends, plus the dense distance matrix helper.

Shape conventions used throughout: n1 is the number of reference points (X), n2 the number of query points (Y), p the embedding dimension and k the
number of neighbours.
"""

import numpy as np
from scipy.spatial import cKDTree

from .config import get_config


def _resolve_workers(n_ref, n_query, n_jobs):
    """Turn n_jobs into a cKDTree workers count.

    None means decide automatically.

    Parameters
    ----------
    n_ref : int
        Number of reference points, n1.
    n_query : int
        Number of query points, n2.
    n_jobs : int or None
        Requested worker count, or None to decide from the problem size.

    Returns
    -------
    workers : int
        Worker count to hand to cKDTree.query.
    """
    if n_jobs is not None:
        return n_jobs
    if n_ref * n_query < get_config("parallel_min_work"):
        return 1
    return -1


def tree_query(X, Y, k=1, return_distance=False, n_jobs=None, leaf_size=None, **_):
    """Nearest neighbours via a kd-tree. Best in low dimension.

    Trailing ``**_`` is to ignore any extra arguments passed by the dispatcher.

    Parameters
    ----------
    X : np.ndarray
        (n1, p). Reference points, the set being searched.
    Y : np.ndarray
        (n2, p). Query points.
    k : int
        Number of neighbours.
    return_distance : bool
        Whether to also return distances.
    n_jobs : int or None
        -1 uses all cores, 1 forces serial. None (default) uses all cores when
        n1 * n2 >= parallel_min_work, serial below it.
    leaf_size : int, optional
        cKDTree leafsize. Defaults to the configured value.

    Returns
    -------
    dists : np.ndarray, optional
        (n2,) if k = 1 else (n2, k). Distance to each neighbour. Only if return_distance.
    matches : np.ndarray
        (n2,) if k = 1 else (n2, k). Index in X of each neighbour.
    """
    leaf_size = get_config("leaf_size") if leaf_size is None else leaf_size
    workers = _resolve_workers(X.shape[0], Y.shape[0], n_jobs)

    tree = cKDTree(X, leafsize=leaf_size)
    # scipy already squeezes the last axis when k == 1
    dists, matches = tree.query(Y, k=k, workers=workers)  # (n2,) or (n2, k)

    if return_distance:
        return dists, matches
    return matches


def brute_query(X, Y, k=1, return_distance=False, n_jobs=None, working_memory=None, **_):
    r"""Nearest neighbours by brute force. Best above a handful of dimensions.

    Scikit learn uses nice mixed precision with memory handling.

    Parameters
    ----------
    X : np.ndarray
        (n1, p). Reference points, the set being searched.
    Y : np.ndarray
        (n2, p). Query points.
    k : int
        Number of neighbours.
    return_distance : bool
        Whether to also return distances.
    n_jobs : int
        Passed through to scikit-learn.
    working_memory : int, optional
        Chunking budget in MB. Defaults to the configured value.

    Returns
    -------
    dists : np.ndarray, optional
        (n2,) if k = 1 else (n2, k). Distance to each neighbour. Only if return_distance.
    matches : np.ndarray
        (n2,) if k = 1 else (n2, k). Index in X of each neighbour.
    """
    # Pure NumPy implementation with chunking to avoid loading blocked sklearn DLLs
    n1, p = X.shape
    n2 = Y.shape[0]

    # Batch over Y to respect working_memory
    chunk_size = max(1, min(n2, 2048))
    matches_list = []
    dists_list = []

    X_norm_sq = np.sum(X**2, axis=1)

    for i in range(0, n2, chunk_size):
        Y_chunk = Y[i : i + chunk_size]
        Y_norm_sq = np.sum(Y_chunk**2, axis=1)
        # (chunk_size, n1) squared distance matrix
        d2 = Y_norm_sq[:, None] + X_norm_sq[None, :] - 2 * (Y_chunk @ X.T)
        np.maximum(d2, 0, out=d2)

        if k == 1:
            idx = np.argmin(d2, axis=1)
            matches_list.append(idx)
            if return_distance:
                dists_list.append(np.sqrt(d2[np.arange(len(idx)), idx]))
        else:
            idx = np.argpartition(d2, k, axis=1)[:, :k]
            # sort the k nearest
            row_idx = np.arange(len(idx))[:, None]
            part_dists = d2[row_idx, idx]
            sort_order = np.argsort(part_dists, axis=1)
            sorted_idx = np.take_along_axis(idx, sort_order, axis=1)
            matches_list.append(sorted_idx)
            if return_distance:
                sorted_dists = np.sqrt(np.take_along_axis(part_dists, sort_order, axis=1))
                dists_list.append(sorted_dists)

    matches = np.concatenate(matches_list, axis=0)
    if return_distance:
        dists = np.concatenate(dists_list, axis=0)
        return dists, matches
    return matches


def compute_sqdistmat(X, Y, normalized=False):
    """Pairwise squared Euclidean distance matrix between two sets of points X and Y.

    Parameters
    ----------
    X : np.ndarray
        (n1, p) The first set of points.
    Y : np.ndarray
        (n2, p) The second set of points.
    normalized : bool
        Whether the points already have unit norm. If so the squared distance reduces
        to 2 - 2 X.Y, which skips the two norm computations.

    Returns
    -------
    distmat : np.ndarray
        (n1, n2). Squared Euclidean distance between each pair.
    """

    if not normalized:
        # (n1, 1) + (1, n2) -> (n1, n2)
        return (
            np.square(X).sum(-1, keepdims=True)
            + np.square(Y).sum(-1, keepdims=True).T
            - 2 * (X @ Y.T)
        )
    else:
        return 2 - 2 * X @ Y.T  # (n1, n2)
