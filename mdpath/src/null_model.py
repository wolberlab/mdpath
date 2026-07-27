"""Block Reference Analysis --- :mod:`mdpath.src.null_model`
==============================================================================

This module contains the class :class:`BlockSplitNull` which builds a per-edge
**within-source block reference** (a finite-sampling *stability* reference) for
MDPath contact edges from contiguous, non-overlapping blocks of the source
trajectories.

**Terminology.** This is deliberately *not* an external "no coupling" or "no
reproducibility" generative null: the blocks are windows of the real
trajectories and therefore still contain genuine conformational changes, slow
transitions, estimator noise, temporal dependence, and any drift. The procedure
estimates the *within-trajectory sampling variability* of the edge stability
score, so its outputs are best read as a sampling-stability reference rather than
a formal null distribution.

Per-block and matched-observed values are likewise only comparable under a **fixed
histogram support** (``hist_range``); with ``None`` each block derives its own bin
edges and the block-to-block difference in binning is absorbed into the reference as
if it were sampling noise.

The design sweeps block length ``L`` (ns) and replica count ``R`` so every edge
is compared at the *same* ``(L, R)`` as the matched observed estimate. For each
source and ``(L, R)`` the reference is estimated by **resampling** ``draws``
random groups of ``R`` blocks (see :func:`_source_task`), summarised per edge by
the median stability score ``c_null_med`` and an across-source inter-quartile
band (``c_null_iqr_lo``/``c_null_iqr_hi``, across sources). The finite-replica
lower bound is ``c_min_R`` (:func:`confidence_floor`); the dispersion ratio
``rho`` = cv_obs / cv_null is the directly interpretable comparison.

``c_corr_experimental`` is an **experimental heuristic** (quadrature CV
subtraction) whose additive-error assumptions are not established here. Its state
is reported categorically in ``corr_status`` (``undefined`` for nonfinite CVs,
``below_or_equal_reference`` at or under the reference, ``above_reference`` only
when strictly above); the score is finite **only** for ``above_reference`` and is
``NaN`` (unresolved) otherwise -- never mapped to perfect confidence.

Draws are design-point-specific: for each ``(source, L, R)`` the block groups are
resampled with an independent generator (and enumerated exactly when few groups
exist), so a fixed ``(data, source, L, R, seed)`` is invariant to the rest of the
sweep. Outputs are the reproducible CSVs ``edge_null_confidence``,
``null_by_source``, ``null_by_length``, ``matched_comparison`` and
``null_edge_by_source`` (per-source detail incl. the within-source draw IQR),
plus ``run_metadata.json``.

Classes
--------

:class:`BlockSplitNull`

Functions
---------

:func:`confidence_floor`
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from mdpath.src.edge_utils import canonical_pair
from mdpath.src.mutual_information import NMICalculator

Pair = Tuple[int, int]


def _edge_vector(
    frame: pd.DataFrame,
    pairs: Sequence[Pair],
    num_bins: int,
    hist_range=(-180.0, 180.0),
) -> np.ndarray:
    """Computes the per-edge mutual-information difference for a single trajectory block.

    Runs :class:`~mdpath.src.mutual_information.NMICalculator` on ``frame`` for the
    requested residue pairs and returns their ``MI Difference`` values in the same
    order as ``pairs``, filling any pair the calculator did not return with ``NaN``.

    Args:
        frame (pd.DataFrame): Dihedral data for a single block of frames.

        pairs (Sequence[Pair]): Ordered residue pairs to evaluate.

        num_bins (int): Number of histogram bins for the mutual information calculator.

        hist_range (tuple | None): Fixed ``(low, high)`` histogram support. Blocks are
            only comparable to each other, and to the matched observed windows, under a
            shared support; ``None`` restores per-block data-derived binning.

    Returns:
        np.ndarray: One MI-difference value per requested pair, ordered like ``pairs``.
    """
    result = NMICalculator(
        frame,
        num_bins=num_bins,
        pairs=pairs,
        show_progress=False,
        hist_range=hist_range,
    ).nmi_df

    lookup = {
        canonical_pair(pair): value
        for pair, value in zip(result["Residue Pair"], result["MI Difference"])
    }
    return np.asarray([lookup.get(pair, np.nan) for pair in pairs], dtype=float)


def confidence_floor(n_replicas: int, ddof: int = 0) -> float:
    """Returns the finite-replica lower bound for the confidence ``1 / (1 + CV)``.

    With a finite number of replicas the coefficient of variation
    ``CV = sigma / mu`` cannot exceed ``sqrt((R - 1) * R / (R - ddof))``, so the
    confidence ``1 / (1 + CV)`` has a corresponding lower bound. This floor is
    reported per matched replica count as the ``c_min_R`` column.

    Args:
        n_replicas (int): Number of replicas ``R`` (must be at least two).

        ddof (int): Delta degrees of freedom for the standard deviation (0 or 1).

    Returns:
        float: The finite-replica lower bound on the confidence ``1 / (1 + CV)``.

    Raises:
        ValueError: If ``n_replicas`` is below two or ``ddof`` is not 0 or 1.
    """
    if n_replicas < 2 or ddof not in (0, 1):
        raise ValueError("n_replicas must be >= 2 and ddof must be 0 or 1")
    cv_max = np.sqrt((n_replicas - 1) * n_replicas / (n_replicas - ddof))
    return float(1.0 / (1.0 + cv_max))


def _draw_summary(
    weights: np.ndarray,
    groups: Sequence[np.ndarray],
    ddof: int,
    max_cells: int = 4_000_000,
) -> Mapping[str, np.ndarray]:
    """Summarises the per-draw statistics of resampled block groups.

    Gathers the sampled groups into a ``(draws, R, edges)`` cube and reduces along
    the group axis in one vectorised pass rather than looping over draws. The cube
    is materialised in chunks so peak memory stays bounded (``max_cells`` elements)
    regardless of the draw, replica and edge counts; chunking cannot change the
    result because every draw is reduced independently.

    Args:
        weights (np.ndarray): Per-block edge weights, ``(n_blocks, n_edges)``.

        groups (Sequence[np.ndarray]): Sampled block-index groups, each of length ``R``.

        ddof (int): Delta degrees of freedom for the standard deviation (0 or 1).

        max_cells (int): Soft cap on the number of cube elements held at once.

    Returns:
        Mapping[str, np.ndarray]: Per-edge across-draw medians under ``"mu"``,
        ``"sigma"``, ``"cv"`` and ``"c"``, plus the draw inter-quartile bounds of
        the confidence (``"c_lo"``/``"c_hi"``) and of the coefficient of variation
        (``"cv_lo"``/``"cv_hi"``, which propagate into the ``rho`` interval).
    """
    index = np.asarray(groups)
    n_draws, r_null = index.shape
    n_edges = weights.shape[1]
    chunk = max(1, int(max_cells // max(1, r_null * n_edges)))
    mu = np.empty((n_draws, n_edges), dtype=float)
    sigma = np.empty((n_draws, n_edges), dtype=float)
    for start in range(0, n_draws, chunk):
        cube = weights[index[start : start + chunk]]
        mu[start : start + chunk] = np.nanmean(cube, axis=1)
        sigma[start : start + chunk] = np.nanstd(cube, axis=1, ddof=ddof)
    cv = np.divide(sigma, mu, out=np.full_like(mu, np.nan), where=mu > 0)
    c = 1.0 / (1.0 + cv)
    c_lo, c_hi = np.nanquantile(c, [0.25, 0.75], axis=0)
    cv_lo, cv_hi = np.nanquantile(cv, [0.25, 0.75], axis=0)
    return {
        "mu": np.nanmedian(mu, axis=0),
        "sigma": np.nanmedian(sigma, axis=0),
        "cv": np.nanmedian(cv, axis=0),
        "c": np.nanmedian(c, axis=0),
        "c_lo": c_lo,
        "c_hi": c_hi,
        "cv_lo": cv_lo,
        "cv_hi": cv_hi,
    }


def _source_task(payload: tuple) -> Tuple[pd.DataFrame, List[dict]]:
    """Builds the block-reference edge and QC records for a single source trajectory.

    Splits the source into non-overlapping blocks at every requested block length,
    computes each block's per-edge weights once, and estimates the reference edge
    statistics by **resampling** ``draws`` random groups of ``R_null`` blocks from
    all available blocks (without replacement within a group), reusing the same
    per-block weights across draws. Every edge's per-source reference value is the
    median of its resampling distribution; a within-source draw inter-quartile
    range is retained. When ``R_null`` equals the number of available blocks the
    only feasible group is the full set, so the resampling collapses to that set.

    Args:
        payload (tuple): Packed arguments ``(source_id, source, lengths, r_values,
            pairs, num_bins, ddof, draws, seed, hist_range)`` so the task can run
            inside a worker process.

    Returns:
        Tuple[pd.DataFrame, List[dict]]: The per-edge reference table (one row per
        edge, block length and ``R_null``; built columnwise to stay cheap for the
        millions of rows a long multi-source sweep produces) and the per-source QC
        records carrying the across-edge median/IQR of ``c_null`` plus block and
        draw bookkeeping.
    """
    (
        source_id,
        source,
        lengths,
        r_values,
        pairs,
        num_bins,
        ddof,
        draws,
        seed,
        hist_range,
    ) = payload
    n_edges = len(pairs)
    res_u = np.fromiter((pair[0] for pair in pairs), dtype=int, count=n_edges)
    res_v = np.fromiter((pair[1] for pair in pairs), dtype=int, count=n_edges)
    edge_blocks: List[dict] = []
    qc_records: List[dict] = []
    for block_ns, block_frames in lengths:
        n_available = len(source) // block_frames
        blocks = [
            source.iloc[index * block_frames : (index + 1) * block_frames].reset_index(
                drop=True
            )
            for index in range(n_available)
        ]
        weights = np.vstack(
            [_edge_vector(block, pairs, num_bins, hist_range) for block in blocks]
        )
        for r_null in r_values[block_ns]:
            groups = _block_groups(
                n_available, r_null, draws, seed, source_id, block_frames
            )
            n_unique = _count_unique_groups(groups)
            summary = _draw_summary(weights, groups, ddof)
            c_med = summary["c"]
            edge_blocks.append(
                {
                    "source_id": np.full(n_edges, source_id, dtype=int),
                    "block_len_ns": np.full(n_edges, block_ns, dtype=float),
                    "R_null": np.full(n_edges, r_null, dtype=int),
                    "res_u": res_u,
                    "res_v": res_v,
                    "mu_null": summary["mu"],
                    "sigma_null": summary["sigma"],
                    "cv_null": summary["cv"],
                    "c_null": c_med,
                    "c_null_draw_iqr_lo": summary["c_lo"],
                    "c_null_draw_iqr_hi": summary["c_hi"],
                    "n_blocks_available": np.full(n_edges, n_available, dtype=int),
                    "n_groups": np.full(n_edges, len(groups), dtype=int),
                    "n_unique_draws": np.full(n_edges, n_unique, dtype=int),
                }
            )
            q25, q75 = np.nanquantile(c_med, [0.25, 0.75])
            qc_records.append(
                {
                    "source_id": source_id,
                    "block_len_ns": block_ns,
                    "n_blocks_available": n_available,
                    "n_groups": len(groups),
                    "n_unique_draws": n_unique,
                    "R_null": r_null,
                    "median_c_null": np.nanmedian(c_med),
                    "iqr_c_null": q75 - q25,
                    "iqr_c_null_lo": q25,
                    "iqr_c_null_hi": q75,
                    "discarded_frames": len(source) - n_available * block_frames,
                    "ddof": ddof,
                }
            )
    edge_frame = pd.DataFrame(
        {
            column: np.concatenate([block[column] for block in edge_blocks])
            for column in edge_blocks[0]
        }
    )
    return edge_frame, qc_records


def _block_groups(
    n_available, r_null, draws, seed, namespace, tag, enumerate_small=True
):
    """Return the block-index groups sampled for one design point.

    Uses a **design-point-specific** random generator so that a fixed
    ``(data, source, L, R, seed)`` is invariant to the rest of the sweep (adding
    another block length or replica count, or reordering work, does not change
    which groups are sampled here). When the number of possible groups
    ``C(n_available, R)`` does not exceed ``draws`` every distinct group is
    enumerated exactly (no duplicate draws); otherwise ``draws`` groups are drawn
    without replacement within each group.

    Args:
        n_available (int): Number of non-overlapping blocks available.
        r_null (int): Number of blocks per group.
        draws (int): Requested number of random draws.
        seed (int): Base seed.
        namespace (int): Namespace component (e.g. source id, or a fixed tag for
            the observed side) keeping generators independent.
        tag (int): Design-point component (block length in frames).
        enumerate_small (bool): Enumerate every combination when the space is
            smaller than ``draws``. Valid only when a draw is fully determined by
            the chosen indices; the observed side with random windows must set this
            ``False``, because there a draw also carries a window choice and
            collapsing to the few distinct index sets would discard that variation.

    Returns:
        list[np.ndarray]: The sampled block-index groups.
    """
    if enumerate_small and comb(n_available, r_null) <= draws:
        return [np.asarray(group) for group in combinations(range(n_available), r_null)]
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(namespace), int(tag), int(r_null)])
    )
    return [rng.choice(n_available, size=r_null, replace=False) for _ in range(draws)]


def _count_unique_groups(groups) -> int:
    """Count distinct block *sets* among sampled groups.

    A group is an unordered set of block indices, so ``(1, 2)`` and ``(2, 1)`` are
    the same draw. Each group is canonicalised (sorted) before counting to avoid
    over-reporting the effective design coverage.

    Args:
        groups (Iterable): Sampled block-index groups.

    Returns:
        int: Number of distinct sorted groups.
    """
    return len({tuple(sorted(int(i) for i in group)) for group in groups})


def _observed_task(payload: tuple) -> Dict[float, np.ndarray]:
    """Computes the observed per-edge weights for one replica at every block length.

    With ``observed_window="random"`` (the default) the replica is split into the
    same contiguous, non-overlapping windows the reference uses, and every window
    is evaluated; :meth:`BlockSplitNull.run` then draws one window per selected
    replica. The legacy ``"leading"`` mode evaluates only the first window, which
    uses a small, fixed slice of each replica and is kept solely to reproduce runs
    made before the observed side was randomised.

    Args:
        payload (tuple): Packed arguments ``(frame, lengths, pairs, num_bins,
            observed_window, hist_range)`` so the task can run inside a worker process.

    Returns:
        Dict[float, np.ndarray]: Mapping from block length in nanoseconds to a
        ``(n_windows, n_edges)`` weight matrix for this replica (a single row in
        ``"leading"`` mode).
    """
    frame, lengths, pairs, num_bins, observed_window, hist_range = payload
    result: Dict[float, np.ndarray] = {}
    for block_ns, block_frames in lengths:
        n_windows = 1 if observed_window == "leading" else len(frame) // block_frames
        result[block_ns] = np.vstack(
            [
                _edge_vector(
                    frame.iloc[
                        index * block_frames : (index + 1) * block_frames
                    ].reset_index(drop=True),
                    pairs,
                    num_bins,
                )
                for index in range(n_windows)
            ]
        )
    return result


def _observed_row_groups(n_windows, replica_groups, seed, tag, r_null):
    """Pick one random window per selected replica for every draw.

    Mirrors how the reference draws its blocks, so the matched observed estimate
    is built from windows sampled across the whole trajectory rather than a fixed
    leading slice. Rows index into the replicas' windows stacked end to end.

    The generator takes a five-part entropy tuple, which cannot collide with the
    four-part tuples :func:`_block_groups` uses, so replica selection and window
    selection stay independent while both remain design-point specific.

    Args:
        n_windows (np.ndarray): Number of windows available per replica.
        replica_groups (Sequence[np.ndarray]): The replicas selected for each draw.
        seed (int): Base seed.
        tag (int): Design-point component (block length in frames).
        r_null (int): Replicas per draw.

    Returns:
        list[np.ndarray]: One row-index group per draw.
    """
    offsets = np.concatenate([[0], np.cumsum(n_windows)[:-1]])
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0, int(tag), int(r_null), 1])
    )
    return [
        offsets[group] + rng.integers(0, n_windows[group]) for group in replica_groups
    ]


class BlockSplitNull:
    """Estimate edge-specific stability references from within-source blocks.

    Every reference estimate uses blocks from exactly one source; source-level
    estimates are aggregated only after their edge statistics have been formed.
    For each source and ``(L, R)`` the reference is estimated by resampling
    ``draws`` random groups of ``R`` blocks, and the matched observed estimate
    resamples ``draws`` random groups of ``R`` observed replicas at the same ``L``.

    Attributes:
        sources (List[pd.DataFrame]): Source trajectories the reference blocks are drawn from.

        observed (List[pd.DataFrame]): Observed replica trajectories used for the matched subsampling.

        frames_per_ns (float): Sampling density used to convert block lengths (ns) to frames.

        lengths (List[Tuple[float, int]]): Feasible block lengths as ``(length_ns, block_frames)`` pairs.

        pairs (List[Pair]): Normalised contact residue pairs present in every trajectory.

        num_bins (int): Number of histogram bins passed to the MI calculator.

        hist_range (tuple | None): Fixed ``(low, high)`` histogram support passed to the
            MI calculator. Per-block and matched-observed values are only comparable under
            a shared support; ``None`` restores per-input data-derived binning and is
            retained only to reproduce pre-fix numbers.

        draws (int): Number of random resampling draws for both the block reference and the matched observed estimate.

        seed (int): Seed for the resampling random generators.

        ddof (int): Delta degrees of freedom for the standard deviation (0 or 1).

        n_jobs (int): Maximum number of worker processes used for the block and observed maps.

        observed_window (str): How each selected replica contributes its window of
            length ``L``. ``"random"`` (default) draws a random one of the replica's
            non-overlapping windows, mirroring the reference and using the whole
            trajectory. ``"leading"`` always takes the first window; it is retained
            only to reproduce runs made before the observed side was randomised and
            is biased whenever the start of a trajectory is atypical.
    """

    def __init__(
        self,
        source_dfs: Sequence[pd.DataFrame],
        observed_dfs: Sequence[pd.DataFrame],
        block_lengths_ns: Sequence[float] = (25.0, 50.0, 100.0, 250.0),
        frames_per_ns: float = 1.0,
        pairs: Iterable[Sequence[object]] | None = None,
        num_bins: int = 35,
        draws: int = 100,
        seed: int = 0,
        ddof: int = 0,
        n_jobs: int = 1,
        observed_window: str = "random",
        hist_range=(-180.0, 180.0),
    ) -> None:
        if len(source_dfs) < 1 or len(observed_dfs) < 2:
            raise ValueError(
                "block null needs a source and at least two observed replicas"
            )
        if frames_per_ns <= 0 or draws < 1 or ddof not in (0, 1):
            raise ValueError(
                "frames_per_ns and draws must be positive; ddof must be 0 or 1"
            )
        if observed_window not in ("random", "leading"):
            raise ValueError("observed_window must be 'random' or 'leading'")
        columns = set(source_dfs[0].columns)
        for frame in list(source_dfs[1:]) + list(observed_dfs):
            columns &= set(frame.columns)
        available = {int(str(column).split()[-1]) for column in columns}
        candidate = (canonical_pair(pair) for pair in (pairs or ()) if len(pair) == 2)
        normalized = sorted(
            {
                edge
                for edge in candidate
                if edge[0] != edge[1] and set(edge) <= available
            }
        )
        if not normalized:
            raise ValueError(
                "no requested contact edges are present in every trajectory"
            )
        requested = sorted({float(value) for value in block_lengths_ns})
        if not requested or min(requested) <= 0:
            raise ValueError("block lengths must be positive")
        self.sources = list(source_dfs)
        self.observed = list(observed_dfs)
        self.frames_per_ns = float(frames_per_ns)
        self.pairs = normalized
        self.num_bins = num_bins
        self.draws, self.seed, self.ddof = int(draws), int(seed), ddof
        self.n_jobs = max(1, int(n_jobs))
        self.observed_window = observed_window
        self.hist_range = hist_range

        # Feasibility is evaluated per length: a length is usable only if every
        # source yields at least two non-overlapping blocks and every observed
        # replica is at least one block long. Infeasible lengths are dropped with
        # a warning rather than aborting the whole run (the run() method still
        # requires the chosen primary length to survive).
        min_source = min(map(len, self.sources))
        min_observed = min(map(len, self.observed))
        self.lengths: List[Tuple[float, int]] = []
        dropped: List[float] = []
        for value in requested:
            block_frames = max(1, int(round(value * self.frames_per_ns)))
            if min_source // block_frames < 2 or min_observed < block_frames:
                dropped.append(value)
                continue
            self.lengths.append((value, block_frames))
        if dropped:
            warnings.warn(
                "block-reference: dropping infeasible block length(s) "
                f"{', '.join(f'{value:g} ns' for value in dropped)} "
                f"(each source has >= {min_source} frames at "
                f"{self.frames_per_ns:g} frames/ns; a length needs two blocks per source).",
                stacklevel=2,
            )
        if not self.lengths:
            raise ValueError(
                "no requested block length is feasible: every source needs at least "
                "two blocks and every observed replica at least one block."
            )

    @staticmethod
    def split_blocks(
        frame: pd.DataFrame, block_length: int
    ) -> Tuple[List[pd.DataFrame], int]:
        """Splits a trajectory into non-overlapping blocks and reports the discarded remainder.

        Args:
            frame (pd.DataFrame): Trajectory frames to split.

            block_length (int): Number of frames per block (must be positive).

        Returns:
            Tuple[List[pd.DataFrame], int]: The list of contiguous blocks and the
            number of trailing frames discarded because they do not fill a whole block.

        Raises:
            ValueError: If ``block_length`` is less than one.
        """
        if block_length < 1:
            raise ValueError("block_length must be positive")
        n_blocks = len(frame) // block_length
        blocks = [
            frame.iloc[index * block_length : (index + 1) * block_length].reset_index(
                drop=True
            )
            for index in range(n_blocks)
        ]
        return blocks, len(frame) - n_blocks * block_length

    def _design(self) -> Tuple[Dict[float, List[int]], Dict[float, int], int]:
        """Builds the feasible 2-D sweep grid of replica counts per block length.

        For each block length the largest matched replica count is the smaller of
        the number of observed replicas and the fewest blocks any source yields;
        the grid enumerates every ``R`` from two up to that value so cross-system
        comparisons at an explicitly shared ``(L, R)`` remain possible.

        Returns:
            Tuple[Dict[float, List[int]], Dict[float, int], int]: The per-length
            sweep grid mapping block length to its list of replica counts, the
            per-length native (maximum) replica count, and the smallest native
            count across all lengths (the fixed ``R`` used for balanced comparisons).

        Raises:
            ValueError: If any sweep length supports fewer than two matched blocks.
        """
        native = {
            ns: min(
                len(self.observed),
                min(len(source) // frames for source in self.sources),
            )
            for ns, frames in self.lengths
        }
        if min(native.values()) < 2:
            raise ValueError(
                "every sweep length must support at least two matched blocks"
            )
        
        grid = {ns: list(range(2, native[ns] + 1)) for ns, _ in self.lengths}
        return grid, native, min(native.values())

    def _map(self, function, payloads):
        """Applies ``function`` to each payload, serially or across a process pool.

        Args:
            function (Callable): Worker callable applied to a single payload.

            payloads (Sequence): Payloads to process.

        Returns:
            list: Results in payload order. A process pool is used only when
            ``n_jobs`` is greater than one and there is more than one payload.
        """
        if self.n_jobs == 1 or len(payloads) == 1:
            return [function(payload) for payload in payloads]
        with ProcessPoolExecutor(
            max_workers=min(self.n_jobs, len(payloads))
        ) as executor:
            return list(executor.map(function, payloads))

    @staticmethod
    def write_csv(frame: pd.DataFrame, path: Path) -> None:
        """Writes a DataFrame to CSV in a stable, byte-reproducible form.

        Args:
            frame (pd.DataFrame): Table to serialise.

            path (Path): Destination CSV path.
        """
        frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")

    def run(
        self, outdir: str | Path, primary_block_ns: float = 100.0
    ) -> Dict[str, pd.DataFrame]:
        """Runs the full block-reference grid and matched observed subsampling, then writes the CSVs.

        Splits every source into blocks over the requested length grid, aggregates
        the per-edge reference stability score (median ``c_null_med`` with the
        across-source band ``c_null_iqr_lo``/``c_null_iqr_hi``), draws matched
        observed subsamples to obtain the observed stability, and combines them
        into the dispersion ratio ``rho`` and the experimental corrected score
        ``c_corr_experimental`` (finite only when ``corr_status == 'above_reference'``)
        at the finite-replica lower bound ``c_min_R``. Five CSVs (and
        ``run_metadata.json``) are written to ``outdir``: ``edge_null_confidence``
        (per-edge matched observed vs reference at the primary design point),
        ``null_by_source`` (per-source QC), ``null_by_length`` (reference trend
        across block length and replica count), ``matched_comparison`` (the full
        observed-versus-reference grid) and ``null_edge_by_source`` (per-source,
        per-edge detail incl. the within-source draw IQR).

        Args:
            outdir (str | Path): Directory the CSV outputs are written to (created if absent).

            primary_block_ns (float): Block length in nanoseconds used for the
                per-edge ``edge_null_confidence`` table; must also appear in
                ``block_lengths_ns``.

        Returns:
            Dict[str, pd.DataFrame]: Mapping from output name
            (``"edge_null_confidence"``, ``"null_by_source"``, ``"null_by_length"``,
            ``"matched_comparison"``, ``"null_edge_by_source"``) to the written DataFrame.

        Raises:
            ValueError: If ``primary_block_ns`` is not one of the feasible block lengths.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        grid, native, fixed_r = self._design()
        valid_lengths = {ns for ns, _ in self.lengths}
        if float(primary_block_ns) not in valid_lengths:
            raise ValueError(
                f"primary block length {float(primary_block_ns):g} ns is not feasible; "
                f"feasible lengths are {sorted(valid_lengths)}. Choose a shorter primary "
                "length or provide longer trajectories."
            )
        
        print(
            "Block-reference design: %g frames/ns, %d source(s), %d observed replica(s), "
            "%d contact edge(s), %d draw(s)."
            % (
                self.frames_per_ns,
                len(self.sources),
                len(self.observed),
                len(self.pairs),
                self.draws,
            )
        )
        for block_ns, block_frames in self.lengths:
            print(
                "  L=%g ns (%d frames/block): up to R=%d matched blocks, grid R in [%d, %d]."
                % (
                    block_ns,
                    block_frames,
                    native[block_ns],
                    min(grid[block_ns]),
                    max(grid[block_ns]),
                )
            )

        source_payloads = [
            (
                source_id + 1,
                source,
                self.lengths,
                grid,
                self.pairs,
                self.num_bins,
                self.ddof,
                self.draws,
                self.seed,
                self.hist_range,
            )
            for source_id, source in enumerate(self.sources)
        ]
        source_results = self._map(_source_task, source_payloads)
        source_edges = pd.concat(
            [edges for edges, _ in source_results], ignore_index=True
        )
        null_by_source = pd.DataFrame(
            [row for _, result in source_results for row in result]
        )

        grouped = source_edges.groupby(
            ["block_len_ns", "R_null", "res_u", "res_v"], sort=True
        )
        null_edges = grouped.agg(
            mu_null_med=("mu_null", "median"),
            sigma_null_med=("sigma_null", "median"),
            cv_null_med=("cv_null", "median"),
            c_null_med=("c_null", "median"),
            c_null_iqr_lo=("c_null", lambda values: values.quantile(0.25)),
            c_null_iqr_hi=("c_null", lambda values: values.quantile(0.75)),
            c_null_min=("c_null", "min"),
            c_null_max=("c_null", "max"),
            n_sources=("source_id", "nunique"),
        ).reset_index()

        observed_payloads = [
            (
                frame,
                self.lengths,
                self.pairs,
                self.num_bins,
                self.observed_window,
                self.hist_range,
            )
            for frame in self.observed
        ]
        observed_results = self._map(_observed_task, observed_payloads)
 
        observed_weights = {
            ns: np.vstack([result[ns] for result in observed_results])
            for ns, _ in self.lengths
        }
        observed_n_windows = {
            ns: np.array([len(result[ns]) for result in observed_results])
            for ns, _ in self.lengths
        }
        matched_records: List[dict] = []
        for block_ns, block_frames in self.lengths:
            for r_null in grid[block_ns]:

                replica_groups = _block_groups(
                    len(self.observed),
                    r_null,
                    self.draws,
                    self.seed,
                    0,
                    block_frames,
                    enumerate_small=self.observed_window == "leading",
                )
     
                groups = _observed_row_groups(
                    observed_n_windows[block_ns],
                    replica_groups,
                    self.seed,
                    block_frames,
                    r_null,
                )
                obs = _draw_summary(observed_weights[block_ns], groups, self.ddof)

                null_slice = (
                    null_edges[
                        (null_edges.block_len_ns == block_ns)
                        & (null_edges.R_null == r_null)
                    ]
                    .set_index(["res_u", "res_v"])
                    .reindex(pd.MultiIndex.from_tuples(self.pairs))
                )
                cv_obs = obs["cv"]
                cv_null = null_slice.cv_null_med.to_numpy()
  
                defined = np.isfinite(cv_obs) & np.isfinite(cv_null)
                above = defined & (cv_obs > cv_null)
                cv_gen = np.where(
                    above, np.sqrt(np.where(above, cv_obs**2 - cv_null**2, 0.0)), np.nan
                )
                matched_records.append(
                    {
                        "res_u": np.fromiter(
                            (pair[0] for pair in self.pairs), int, len(self.pairs)
                        ),
                        "res_v": np.fromiter(
                            (pair[1] for pair in self.pairs), int, len(self.pairs)
                        ),
                        "L": np.full(len(self.pairs), block_ns, dtype=float),
                        "R_null": np.full(len(self.pairs), r_null, dtype=int),
                        "L_obs": np.full(len(self.pairs), block_ns, dtype=float),
                        "R_obs": np.full(len(self.pairs), r_null, dtype=int),
                        "mu_obs_matched_med": obs["mu"],
                        "sigma_obs_matched_med": obs["sigma"],
                        "cv_obs_matched_med": cv_obs,
                        "c_obs_matched_med": obs["c"],
                        "c_obs_matched_iqr": obs["c_hi"] - obs["c_lo"],
                        "c_obs_matched_iqr_lo": obs["c_lo"],
                        "c_obs_matched_iqr_hi": obs["c_hi"],
                        "c_null_med": null_slice.c_null_med.to_numpy(),
                        "c_null_iqr_lo": null_slice.c_null_iqr_lo.to_numpy(),
                        "c_null_iqr_hi": null_slice.c_null_iqr_hi.to_numpy(),
                        "rho": np.divide(
                            cv_obs,
                            cv_null,
                            out=np.full(len(self.pairs), np.nan),
                            where=np.isfinite(cv_null) & (cv_null > 0),
                        ),
     
                        "rho_iqr_lo": np.divide(
                            obs["cv_lo"],
                            cv_null,
                            out=np.full(len(self.pairs), np.nan),
                            where=np.isfinite(cv_null) & (cv_null > 0),
                        ),
                        "rho_iqr_hi": np.divide(
                            obs["cv_hi"],
                            cv_null,
                            out=np.full(len(self.pairs), np.nan),
                            where=np.isfinite(cv_null) & (cv_null > 0),
                        ),
                        "cv_gen": cv_gen,
                        "c_corr_experimental": np.where(
                            above, 1.0 / (1.0 + cv_gen), np.nan
                        ),
                        "corr_status": np.where(
                            ~defined,
                            "undefined",
                            np.where(
                                above, "above_reference", "below_or_equal_reference"
                            ),
                        ),
                        "observed_below_reference": defined & ~above,
                        "ddof": np.full(len(self.pairs), self.ddof, dtype=int),
                        "c_min_R": np.full(
                            len(self.pairs), confidence_floor(r_null, self.ddof)
                        ),
                    }
                )
        matched = pd.DataFrame(
            {
                column: np.concatenate([block[column] for block in matched_records])
                for column in matched_records[0]
            }
        ).sort_values(["L", "R_null", "res_u", "res_v"])

        length_records = []
        for (block_ns, r_null), values in null_edges.groupby(
            ["block_len_ns", "R_null"], sort=True
        ):
            q25, q75 = np.nanquantile(values.c_null_med, [0.25, 0.75])
            design = (
                "fixed_R"
                if r_null == fixed_r
                else ("native" if r_null == native[block_ns] else "grid")
            )
            length_records.append(
                {
                    "block_len_ns": block_ns,
                    "n_blocks": r_null,
                    "R_null": r_null,
                    "median_c_null": np.nanmedian(values.c_null_med),
                    "iqr_c_null": q75 - q25,
                    "c_null_iqr_lo": q25,
                    "c_null_iqr_hi": q75,
                    "design": design,
                    "ddof": self.ddof,
                    "c_min_R": confidence_floor(r_null, self.ddof),
                }
            )
        null_by_length = pd.DataFrame(length_records)
        self._plateau = self._block_length_plateau(null_by_length)
        null_by_source = null_by_source.merge(
            null_by_length[["block_len_ns", "R_null", "design"]],
            on=["block_len_ns", "R_null"],
            how="left",
        ).sort_values(["block_len_ns", "R_null", "source_id"])

        primary_l = float(primary_block_ns)
        primary_r = native[primary_l]
        primary_null = null_edges[
            (null_edges.block_len_ns == primary_l) & (null_edges.R_null == primary_r)
        ]
        primary_obs = matched[(matched.L == primary_l) & (matched.R_null == primary_r)]

        primary_null = primary_null.drop(
            columns=["c_null_iqr_lo", "c_null_iqr_hi", "c_null_med"], errors="ignore"
        )
        edge_output = primary_obs.merge(
            primary_null, on=["res_u", "res_v"], how="inner"
        ).rename(
            columns={
                "mu_obs_matched_med": "mu_obs",
                "sigma_obs_matched_med": "sigma_obs",
                "cv_obs_matched_med": "cv_obs",
                "c_obs_matched_med": "c_obs",
            }
        )
        edge_output["n_blocks"] = primary_r
        edge_output["block_len_ns"] = primary_l
        edge_output["n_sources"] = len(self.sources)
        wanted = [
            "res_u",
            "res_v",
            "mu_obs",
            "sigma_obs",
            "cv_obs",
            "c_obs",
            "R_obs",
            "mu_null_med",
            "sigma_null_med",
            "cv_null_med",
            "c_null_med",
            "c_null_iqr_lo",
            "c_null_iqr_hi",
            "c_null_min",
            "c_null_max",
            "n_blocks",
            "block_len_ns",
            "n_sources",
            "rho",
            "rho_iqr_lo",
            "rho_iqr_hi",
            "cv_gen",
            "c_corr_experimental",
            "corr_status",
            "observed_below_reference",
            "ddof",
            "c_min_R",
        ]
        edge_output = edge_output[wanted].sort_values(["res_u", "res_v"])

        null_edge_by_source = source_edges.sort_values(
            ["block_len_ns", "R_null", "res_u", "res_v", "source_id"]
        )

        files = {
            "edge_null_confidence": edge_output,
            "null_by_source": null_by_source,
            "null_by_length": null_by_length,
            "matched_comparison": matched,
            "null_edge_by_source": null_edge_by_source,
        }
        for name, frame in files.items():
            self.write_csv(frame, outdir / f"{name}.csv")
        if self._plateau.get("plateaued") is False:
            warnings.warn(
                "block-reference: the median reference still shifts by "
                f"{self._plateau['final_rel_change']:.1%} between the two longest "
                f"block lengths ({self._plateau['lengths_ns'][-2]:g} -> "
                f"{self._plateau['lengths_ns'][-1]:g} ns), so L may not yet exceed "
                "the relevant correlation time. Treat the reference as L-dependent "
                "and, if possible, extend the sweep to longer blocks.",
                stacklevel=2,
            )
        self._write_metadata(
            outdir, grid, native, fixed_r, float(primary_block_ns), primary_r
        )
        return files

    @staticmethod
    def _block_length_plateau(null_by_length):
        """Reports whether the reference has stopped changing with block length.

        A block reference is only trustworthy once ``L`` exceeds the relevant
        correlation time; the tell-tale is that the median reference stops moving
        as ``L`` grows. This measures the relative change across the final (longest)
        step of the fixed-``R`` sweep, so the user is not left to eyeball it.

        Args:
            null_by_length (pd.DataFrame): The per-length summary table.

        Returns:
            dict: ``lengths_ns``, ``median_c_null``, ``final_rel_change`` (``None``
            with fewer than two lengths) and a boolean ``plateaued`` (final relative
            change below 1%).
        """
        trend = null_by_length[null_by_length.design == "fixed_R"].sort_values(
            "block_len_ns"
        )
        lengths = [float(value) for value in trend.block_len_ns]
        medians = [float(value) for value in trend.median_c_null]
        if len(medians) < 2 or not np.isfinite(medians[-1]) or medians[-2] == 0:
            return {
                "lengths_ns": lengths,
                "median_c_null": medians,
                "final_rel_change": None,
                "plateaued": None,
            }
        change = abs(medians[-1] - medians[-2]) / abs(medians[-2])
        return {
            "lengths_ns": lengths,
            "median_c_null": medians,
            "final_rel_change": change,
            "plateaued": bool(change < 0.01),
        }

    def _write_metadata(self, outdir, grid, native, fixed_r, primary_l, primary_r):
        """Write a machine-readable ``run_metadata.json`` describing the run.

        Records the configuration and effective design so the CSV outputs are
        self-describing: block lengths in ns and frames, feasible replica ranges,
        the primary design point, draw count, ddof, edge count, and a short glossary
        of the emitted columns.

        Args:
            outdir (Path): Output directory.
            grid (dict): Per-length list of feasible ``R`` values.
            native (dict): Per-length maximum matched ``R``.
            fixed_r (int): The smallest native ``R`` across lengths.
            primary_l (float): Primary block length in ns.
            primary_r (int): Matched ``R`` at the primary length.
        """
        import json

        metadata = {
            "analysis": "within-source block reference (finite-sampling stability reference)",
            "note": (
                "This is NOT an external no-coupling null. Blocks are contiguous "
                "windows of the same trajectories and retain real dynamics; the "
                "reference estimates within-trajectory sampling variability. "
                "c_corr is an experimental heuristic (quadrature CV subtraction)."
            ),
            "frames_per_ns": self.frames_per_ns,
            "n_sources": len(self.sources),
            "n_observed_replicas": len(self.observed),
            "n_contact_edges": len(self.pairs),
            "draws": self.draws,
            "seed": self.seed,
            "ddof": self.ddof,
            "n_jobs": self.n_jobs,
            "observed_window": self.observed_window,
            "hist_range": list(self.hist_range) if self.hist_range else None,
            "block_lengths": [
                {
                    "L_ns": block_ns,
                    "block_frames": block_frames,
                    "native_R": native[block_ns],
                    "feasible_R": [min(grid[block_ns]), max(grid[block_ns])],
                }
                for block_ns, block_frames in self.lengths
            ],
            "fixed_R": fixed_r,
            "primary_design_point": {"L_obs_ns": primary_l, "R_obs": primary_r},
            "block_length_plateau": getattr(self, "_plateau", None),
            "outputs": {
                "edge_null_confidence.csv": "per-edge matched observed vs reference at the primary (L, R)",
                "null_by_source.csv": "per-source, per-(L, R) across-edge QC (median/IQR of c_null, block/draw counts)",
                "null_by_length.csv": "reference trend across block length and replica count",
                "matched_comparison.csv": "the full observed-vs-reference grid",
                "null_edge_by_source.csv": "detailed per-source, per-edge reference records incl. within-source draw IQR",
            },
            "columns": {
                "c_obs": "matched observed stability score at (L_obs, R_obs)",
                "c_null_med": "median block-reference stability score (across sources)",
                "c_null_iqr_lo/hi": "across-source inter-quartile band of c_null (NOT within-source draw IQR)",
                "c_null_draw_iqr_lo/hi": "within-source resampling-draw IQR of c_null (in null_edge_by_source.csv)",
                "rho": "cv_obs / cv_null (dispersion ratio; the directly interpretable comparison)",
                "c_corr_experimental": "EXPERIMENTAL quadrature-corrected score; finite ONLY when corr_status=='above_reference'",
                "corr_status": "one of undefined | below_or_equal_reference | above_reference",
                "observed_below_reference": "True when corr_status=='below_or_equal_reference'",
                "n_blocks_available / n_groups / n_unique_draws": "blocks per source; groups summarised; unique groups sampled",
                "c_min_R": "finite-replica lower bound on the score at R_obs",
            },
        }
        with open(Path(outdir) / "run_metadata.json", "w") as handle:
            json.dump(metadata, handle, indent=2)
