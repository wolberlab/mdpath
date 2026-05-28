"""Edge Confidence Calculation --- :mod:`mdpath.src.confidence`
===============================================================================

This module contains the class `EdgeConfidenceCalculator` which aggregates
Normalized Mutual Information (NMI) computed independently on each replica
trajectory into per-edge confidence statistics (mean, standard deviation,
coefficient of variation, and a normalized confidence score).

The class is only meaningful in multi-replica mode (``-traj`` with two or more
trajectory files). The main pipeline's graph weights are unaffected — this
module produces additional CSV outputs summarizing how consistently each edge's
mutual information is observed across replicas.

Classes
--------

:class:`EdgeConfidenceCalculator`
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from mdpath.src.mutual_information import NMICalculator


class EdgeConfidenceCalculator:
    """Compute per-edge NMI confidence across replica trajectories.

    For every residue pair that appears in every per-replica NMI DataFrame,
    aggregate the replica MI values into ``MI_mean``, ``MI_std``, ``MI_cv``
    (coefficient of variation) and a normalized ``confidence`` score bounded
    in ``(0, 1]`` via ``1 / (1 + MI_cv)``. Higher ``confidence`` means the
    edge's MI is more consistent across replicas.

    Attributes:
        per_replica_dfs (List[pd.DataFrame]): Per-replica dihedral movement DataFrames.

        num_bins (int): Number of histogram bins used by the underlying
            :class:`mdpath.src.mutual_information.NMICalculator`. Default 35.

        per_replica_nmi_dfs (List[pd.DataFrame]): NMI DataFrame computed per replica.
    """

    def __init__(
        self,
        per_replica_dfs: List[pd.DataFrame],
        num_bins: int = 35,
    ) -> None:
        if len(per_replica_dfs) < 2:
            raise ValueError("EdgeConfidenceCalculator requires at least two replicas.")
        self.per_replica_dfs = per_replica_dfs
        self.num_bins = num_bins
        self.per_replica_nmi_dfs = self._compute_per_replica_nmi()

    def _compute_per_replica_nmi(self) -> List[pd.DataFrame]:
        """Compute NMI DataFrame for each replica independently.

        Returns:
            List[pd.DataFrame]: One NMI DataFrame per replica. Confidence is
            always computed on the raw (non-inverted) NMI so that replicas are
            directly comparable even when the main pipeline uses ``-invert``.
        """
        dfs = []
        for i, df in enumerate(self.per_replica_dfs):
            print(
                f"\033[1mComputing NMI for replica {i + 1}/{len(self.per_replica_dfs)}\033[0m"
            )
            nmi_calc = NMICalculator(df, num_bins=self.num_bins, invert=False)
            dfs.append(nmi_calc.nmi_df)
        return dfs

    @staticmethod
    def _residue_pair_to_ints(pair) -> Tuple[int, int]:
        """Convert a ``("Res X", "Res Y")`` tuple into a canonical ``(min, max)`` int pair."""
        a = int(str(pair[0]).split()[-1])
        b = int(str(pair[1]).split()[-1])
        return (a, b) if a <= b else (b, a)

    def build_edge_confidence_df(self) -> pd.DataFrame:
        """Aggregate per-replica NMI into per-edge statistics.

        Returns:
            pd.DataFrame: One row per residue pair. Columns: ``Residue1``,
            ``Residue2``, ``MI_mean``, ``MI_std``, ``MI_cv``, ``confidence``,
            ``n_replicas``, followed by per-replica ``MI_replica_0``,
            ``MI_replica_1``, ... columns. ``Residue1`` < ``Residue2`` by
            construction.
        """
        n_replicas = len(self.per_replica_nmi_dfs)

        replica_lookups = []
        for nmi_df in self.per_replica_nmi_dfs:
            lookup = {}
            for _, row in nmi_df.iterrows():
                key = self._residue_pair_to_ints(row["Residue Pair"])
                if key not in lookup:
                    lookup[key] = float(row["MI Difference"])
            replica_lookups.append(lookup)

        common_pairs = set(replica_lookups[0].keys())
        for lookup in replica_lookups[1:]:
            common_pairs &= set(lookup.keys())

        records = []
        for pair in sorted(common_pairs):
            values = np.array([lookup[pair] for lookup in replica_lookups], dtype=float)
            mi_mean = float(values.mean())
            mi_std = float(values.std(ddof=0))
            mi_cv = float(mi_std / mi_mean) if mi_mean > 0 else float("nan")
            if np.isnan(mi_cv):
                confidence = float("nan")
            else:
                confidence = float(1.0 / (1.0 + mi_cv))
            record = {
                "Residue1": pair[0],
                "Residue2": pair[1],
                "MI_mean": mi_mean,
                "MI_std": mi_std,
                "MI_cv": mi_cv,
                "confidence": confidence,
                "n_replicas": n_replicas,
            }
            for i, v in enumerate(values):
                record[f"MI_replica_{i}"] = float(v)
            records.append(record)

        columns = [
            "Residue1",
            "Residue2",
            "MI_mean",
            "MI_std",
            "MI_cv",
            "confidence",
            "n_replicas",
        ] + [f"MI_replica_{i}" for i in range(n_replicas)]
        return pd.DataFrame(records, columns=columns)

    @staticmethod
    def filter_to_graph_edges(
        edge_confidence_df: pd.DataFrame, graph: nx.Graph
    ) -> pd.DataFrame:
        """Subset an edge confidence DataFrame to only edges present in ``graph``.

        Args:
            edge_confidence_df (pd.DataFrame): Output of :meth:`build_edge_confidence_df`.
            graph (nx.Graph): Residue interaction graph from
                :class:`mdpath.src.graph.GraphBuilder`.

        Returns:
            pd.DataFrame: Rows filtered to residue pairs that are edges in ``graph``.
        """
        graph_edges = set()
        for u, v in graph.edges():
            a, b = int(u), int(v)
            graph_edges.add((a, b) if a <= b else (b, a))

        mask = [
            (int(r1), int(r2)) in graph_edges
            for r1, r2 in zip(
                edge_confidence_df["Residue1"], edge_confidence_df["Residue2"]
            )
        ]
        return edge_confidence_df.loc[mask].reset_index(drop=True)

    @staticmethod
    def build_top_paths_edge_confidence_df(
        top_pathways: List[List[int]], edge_confidence_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Emit one row per consecutive residue transition in each top path.

        Args:
            top_pathways (List[List[int]]): List of paths (each path is a list
                of residue IDs) as produced by the main pipeline.
            edge_confidence_df (pd.DataFrame): Per-edge confidence DataFrame
                from :meth:`build_edge_confidence_df` (not yet filtered to the
                graph — the full set is used so any path transition can be
                looked up).

        Returns:
            pd.DataFrame: One row per ``(path, consecutive-pair)`` with columns
            ``path_index``, ``position_in_path``, ``Residue1``, ``Residue2``,
            ``MI_mean``, ``MI_std``, ``MI_cv``, ``confidence``, ``n_replicas``.
            ``path_index`` is the path's rank in ``top_pathways`` (0 = highest).
        """
        stats_lookup = {}
        for _, row in edge_confidence_df.iterrows():
            a = int(row["Residue1"])
            b = int(row["Residue2"])
            key = (a, b) if a <= b else (b, a)
            stats_lookup[key] = row

        records = []
        for path_index, path in enumerate(top_pathways):
            for position_in_path in range(len(path) - 1):
                u = int(path[position_in_path])
                v = int(path[position_in_path + 1])
                key = (u, v) if u <= v else (v, u)
                stats = stats_lookup.get(key)
                if stats is None:
                    continue
                records.append(
                    {
                        "path_index": path_index,
                        "position_in_path": position_in_path,
                        "Residue1": key[0],
                        "Residue2": key[1],
                        "MI_mean": float(stats["MI_mean"]),
                        "MI_std": float(stats["MI_std"]),
                        "MI_cv": float(stats["MI_cv"]),
                        "confidence": float(stats["confidence"]),
                        "n_replicas": int(stats["n_replicas"]),
                    }
                )

        columns = [
            "path_index",
            "position_in_path",
            "Residue1",
            "Residue2",
            "MI_mean",
            "MI_std",
            "MI_cv",
            "confidence",
            "n_replicas",
        ]
        return pd.DataFrame(records, columns=columns)
