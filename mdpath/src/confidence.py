"""Edge confidence --- :mod:`mdpath.src.confidence`
==============================================================================

This module contains the class :class:`EdgeConfidenceCalculator` which
quantifies how reproducible each residue-pair edge weight is across independent
replicas. It aggregates the per-replica mutual information differences into a
coefficient of variation and derives a stability score ``1 / (1 + CV)``,
together with a **finite-replica lower bound** ``c_min`` on that score and a
**floor-normalised stability score** ``c_norm``.

These scores measure *consistency of the estimated edge magnitude* across
replicas; they are not calibrated probabilities, significance thresholds, or
statements about effect size (an edge with consistently negligible mutual
information can still score highly).

.. important::

   Per-replica values are only comparable to each other under a **fixed histogram
   support**. With ``hist_range=None`` every replica derives its own bin edges from
   its own extrema, so a single wider excursion in one replica changes that
   replica's binning and the difference propagates directly into ``MI_std`` ->
   ``MI_cv`` -> ``confidence``. The default fixed support exists for this reason.

The finite-sampling block reference and its experimental corrected score live in
:mod:`mdpath.src.null_model`.

Classes
--------

:class:`EdgeConfidenceCalculator`
"""

from typing import List

import networkx as nx
import numpy as np
import pandas as pd

from mdpath.src.edge_utils import canonical_pair
from mdpath.src.mutual_information import NMICalculator


class EdgeConfidenceCalculator:
    """Aggregate independently estimated replica edge weights into stability scores.

    Each replica DataFrame is passed through :class:`~mdpath.src.mutual_information.NMICalculator`
    to obtain per-replica mutual information differences for every residue pair. Pairs common to all
    replicas are summarised by their mean, standard deviation and coefficient of variation, from which
    a stability score and a finite-replica-count correction are derived.

    Attributes:
        per_replica_dfs (List[pd.DataFrame]): Input dihedral/feature DataFrames, one per replica.

        num_bins (int): Number of histogram bins forwarded to the mutual information calculator.

        hist_range (tuple | None): Fixed ``(low, high)`` histogram support forwarded to the
            mutual information calculator. Per-replica values are only comparable under a
            fixed support: with ``None`` each replica derives its own bin edges from its own
            extrema, and that difference flows straight into ``MI_std`` -> ``MI_cv`` ->
            ``confidence``. ``None`` is retained only to reproduce pre-fix numbers.

        pairs: Optional restriction of residue pairs forwarded to the mutual information calculator; ``None`` evaluates all pairs.

        ddof (int): Delta degrees of freedom (0 or 1) used for the standard deviation and the finite-replica floor.

        per_replica_nmi_dfs (List[pd.DataFrame]): Per-replica mutual information difference DataFrames computed on construction.
    """

    def __init__(
        self,
        per_replica_dfs: List[pd.DataFrame],
        num_bins: int = 35,
        pairs=None,
        ddof: int = 0,
        hist_range=(-180.0, 180.0),
    ) -> None:
        if len(per_replica_dfs) < 2:
            raise ValueError("EdgeConfidenceCalculator requires at least two replicas.")
        if ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1")
        self.per_replica_dfs = per_replica_dfs
        self.num_bins = num_bins
        self.pairs, self.ddof = pairs, ddof
        self.hist_range = hist_range
        self.per_replica_nmi_dfs = self._compute_per_replica_nmi()

    def _compute_per_replica_nmi(self) -> List[pd.DataFrame]:
        """Run the mutual information calculator on every replica DataFrame.

        Returns:
            List[pd.DataFrame]: One mutual information difference DataFrame per replica, in input order.
        """
        return [
            NMICalculator(
                frame,
                num_bins=self.num_bins,
                pairs=self.pairs,
                show_progress=False,
                hist_range=self.hist_range,
            ).nmi_df
            for frame in self.per_replica_dfs
        ]

    @staticmethod
    def _lookups(nmi_dfs):
        """Index each replica's mutual information differences by residue pair.

        Args:
            nmi_dfs (List[pd.DataFrame]): Per-replica mutual information difference DataFrames, each with ``"Residue Pair"`` and ``"MI Difference"`` columns.

        Returns:
            list: One dictionary per replica mapping the sorted integer residue pair to its mutual information difference.
        """
        return [
            {
                canonical_pair(pair): float(value)
                for pair, value in zip(frame["Residue Pair"], frame["MI Difference"])
            }
            for frame in nmi_dfs
        ]

    def build_edge_confidence_df(self) -> pd.DataFrame:
        """Summarise the per-replica edge weights into a stability-score DataFrame.

        For every residue pair present in all replicas the mean, standard deviation and coefficient
        of variation of the mutual information difference are computed. The stability score is
        ``1 / (1 + cv)``. A **finite-replica lower bound** ``c_min = 1 / (1 + cv_max)`` with
        ``cv_max = sqrt((R - 1) * R / (R - ddof))`` is the smallest score attainable with ``R``
        nonnegative replica values, and ``c_norm`` affinely rescales the attainable interval
        ``[c_min, 1]`` onto ``[0, 1]`` (clipped to guard against floating-point excursions). Both
        are dispersion-based stability scores, not calibrated probabilities.

        Returns:
            pd.DataFrame: One row per common residue pair with columns ``Residue1``, ``Residue2``,
            ``MI_mean``, ``MI_std``, ``MI_cv``, ``confidence``, ``n_replicas``, ``ddof``, ``c_min``
            (finite-replica lower bound), ``c_norm`` (floor-normalised stability score) and one
            ``MI_replica_<i>`` column per replica holding that replica's mutual information difference.
        """
        n_replicas = len(self.per_replica_nmi_dfs)
        lookups = self._lookups(self.per_replica_nmi_dfs)
        common = set.intersection(*(set(lookup) for lookup in lookups))
        cv_max = np.sqrt((n_replicas - 1) * n_replicas / (n_replicas - self.ddof))
        confidence_min = 1.0 / (1.0 + cv_max)
        records = []
        for first, second in sorted(common):
            values = np.asarray([lookup[(first, second)] for lookup in lookups])
            mean, std = float(values.mean()), float(values.std(ddof=self.ddof))
            cv = std / mean if mean > 0 else np.nan
            record = {
                "Residue1": first,
                "Residue2": second,
                "MI_mean": mean,
                "MI_std": std,
                "MI_cv": cv,
                "confidence": 1.0 / (1.0 + cv) if np.isfinite(cv) else np.nan,
                "n_replicas": n_replicas,
                "ddof": self.ddof,
                "c_min": confidence_min,
            }
            record["c_norm"] = (
                float(
                    np.clip(
                        (record["confidence"] - confidence_min)
                        / (1.0 - confidence_min),
                        0.0,
                        1.0,
                    )
                )
                if np.isfinite(record["confidence"])
                else np.nan
            )
            record.update({f"MI_replica_{i}": value for i, value in enumerate(values)})
            records.append(record)
        columns = [
            "Residue1",
            "Residue2",
            "MI_mean",
            "MI_std",
            "MI_cv",
            "confidence",
            "n_replicas",
            "ddof",
            "c_min",
            "c_norm",
        ] + [f"MI_replica_{i}" for i in range(n_replicas)]
        return pd.DataFrame(records, columns=columns)

    @staticmethod
    def filter_to_graph_edges(
        edge_confidence_df: pd.DataFrame, graph: nx.Graph
    ) -> pd.DataFrame:
        """Restrict an edge confidence DataFrame to residue pairs present in a graph.

        Args:
            edge_confidence_df (pd.DataFrame): Edge confidence DataFrame with ``Residue1`` and ``Residue2`` columns.

            graph (nx.Graph): Residue graph whose edges define the pairs to keep.

        Returns:
            pd.DataFrame: The subset of rows whose residue pair is an edge of the graph, with a reset index.
        """
        edges = {canonical_pair(edge) for edge in graph.edges()}
        mask = [
            (int(first), int(second)) in edges
            for first, second in zip(
                edge_confidence_df["Residue1"], edge_confidence_df["Residue2"]
            )
        ]
        return edge_confidence_df.loc[mask].reset_index(drop=True)

    @staticmethod
    def build_top_paths_edge_confidence_df(top_pathways, edge_confidence_df):
        """Expand edge confidence metrics along the edges of the top pathways.

        Args:
            top_pathways (Iterable): Pathways, each a sequence of residue indices, whose consecutive residues form the edges to annotate.

            edge_confidence_df (pd.DataFrame): Edge confidence DataFrame providing the per-edge metrics keyed by residue pair.

        Returns:
            pd.DataFrame: One row per edge occurrence with columns ``path_index``, ``position_in_path``, ``Residue1``, ``Residue2`` followed by every confidence metric column (all columns except the residue identifiers and the per-replica ``MI_replica_<i>`` columns).
        """
        lookup = {
            canonical_pair((row["Residue1"], row["Residue2"])): row
            for _, row in edge_confidence_df.iterrows()
        }
        metrics = [
            column
            for column in edge_confidence_df.columns
            if column not in {"Residue1", "Residue2"}
            and not column.startswith("MI_replica_")
        ]
        records = []
        for path_index, path in enumerate(top_pathways):
            for position, edge in enumerate(zip(path, path[1:])):
                key = canonical_pair(edge)
                if key in lookup:
                    record = {
                        "path_index": path_index,
                        "position_in_path": position,
                        "Residue1": key[0],
                        "Residue2": key[1],
                    }
                    record.update({metric: lookup[key][metric] for metric in metrics})
                    records.append(record)
        columns = ["path_index", "position_in_path", "Residue1", "Residue2"] + metrics
        return pd.DataFrame(records, columns=columns)
