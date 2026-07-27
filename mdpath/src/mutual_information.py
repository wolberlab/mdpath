"""Mutual Information Calculation --- :mod:`mdpath.src.mutual_information`
===============================================================================

This module contains the class `NMICalculator` which calculates the Normalized Mutual Information (NMI)
for all residue pairs in a given dataset based on the dihedral angle movements over the course of the analysed MD trajectory.


Classes
--------

:class:`NMICalculator`

Functions
---------

:func:`binning_diagnostic`
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

from mdpath.src.edge_utils import residue_id


class NMICalculator:
    """Calculate Normalized Mutual Information (NMI) between dihedral angle movements of residue pairs.

    Attributes:
        df_all_residues (pd.DataFrame): DataFrame containing all residues.

        num_bins (int): Number of bins to use for histogram calculations. Default is 35.

        hist_range (tuple | None): Fixed ``(low, high)`` support for the
            dihedral-movement histograms, in degrees. Default ``(-180, 180)``
            fixes the bin edges so that values computed from different
            replicas, blocks or bootstrap samples share one binning and are
            directly comparable. Passing ``None`` restores the previous
            behaviour, in which each input derived its own support from its
            own extrema.

        pairs (list | None): Optional list of residue pairs restricting which pairs
            are evaluated (each pair is any two residue labels, e.g. ``42`` or
            ``"Res 42"``). When ``None`` every pair is computed as before. This only
            selects which pairs are scored -- it does not change the NMI definition.

        show_progress (bool): Whether to show the tqdm progress bar. Default is True.

        nmi_df (pd.DataFrame): DataFrame containing the mutual information differences. Is calculated using either GMM or histogram method.

        entropy_df (pd.DataFrame): Pandas dataframe with residue and entropy values. Is calculated using either GMM or histogram method.
    """

    def __init__(
        self,
        df_all_residues: pd.DataFrame,
        num_bins: int = 35,
        invert=False,
        pairs=None,
        show_progress: bool = True,
        hist_range=(-180.0, 180.0),
    ) -> None:
        self.df_all_residues = df_all_residues
        self.num_bins = num_bins
        self.invert = invert
        self.pairs = pairs
        self.show_progress = show_progress
        self.hist_range = hist_range
        self.nmi_df, self.entropy_df = self.NMI_calcs()

    def _column_pairs(self, columns):
        """Return the ``(col1, col2)`` pairs to evaluate.

        Args:
            columns (list): DataFrame columns, one per residue.

        Returns:
            list: All unordered column pairs, or -- when :attr:`pairs` is set --
            only those matching a requested residue pair (canonicalised, de-duplicated).
        """
        if self.pairs is None:
            return list(combinations(columns, 2))
        by_id = {residue_id(column): column for column in columns}
        result = []
        seen = set()
        for first, second in self.pairs:
            first_id, second_id = residue_id(first), residue_id(second)
            if first_id == second_id or first_id not in by_id or second_id not in by_id:
                continue
            key = tuple(sorted((first_id, second_id)))
            if key not in seen:
                seen.add(key)
                result.append((by_id[key[0]], by_id[key[1]]))
        return result

    def NMI_calcs(self):
        """Extended Normalized Mutual Information and Entropy calculation."""
        columns = self.df_all_residues.columns.tolist()

        histograms = {}
        entropys = {}
        for col in columns:
            hist, _ = np.histogram(
                self.df_all_residues[col],
                bins=self.num_bins,
                range=self.hist_range,
            )
            histograms[col] = hist
            entropys[col] = entropy(hist)

        normalized_mutual_info = {}
        col_pairs = self._column_pairs(columns)
        for col1, col2 in tqdm(
            col_pairs,
            desc="\033[1mCalculating Normalized Mutual Information\033[0m",
            disable=not self.show_progress,
        ):
            hist_joint, _, _ = np.histogram2d(
                self.df_all_residues[col1],
                self.df_all_residues[col2],
                bins=self.num_bins,
                # np.histogram wants (lo, hi); np.histogram2d wants
                # [(lo, hi), (lo, hi)] and raises otherwise. Both accept None,
                # so the None branch is a true no-op.
                range=(
                    None
                    if self.hist_range is None
                    else [self.hist_range, self.hist_range]
                ),
            )
            mi = mutual_info_score(
                histograms[col1], histograms[col2], contingency=hist_joint
            )
            entropy_col1 = entropys[col1]
            entropy_col2 = entropys[col2]
            nmi = mi / np.sqrt(entropy_col1 * entropy_col2)
            normalized_mutual_info[(col1, col2)] = nmi
            normalized_mutual_info[(col2, col1)] = nmi

        entropy_df = pd.DataFrame(entropys.items(), columns=["Residue", "Entropy"])
        nmi_df = pd.DataFrame(
            normalized_mutual_info.items(), columns=["Residue Pair", "MI Difference"]
        )
        if self.invert and not nmi_df.empty:
            max_nmi_diff = nmi_df["MI Difference"].max()
            nmi_df["MI Difference"] = max_nmi_diff - nmi_df["MI Difference"]
        return nmi_df, entropy_df


def binning_diagnostic(
    df_all_residues: pd.DataFrame,
    num_bins: int = 35,
    hist_range=(-180.0, 180.0),
    csv_path: str = "binning_diagnostic.csv",
) -> pd.DataFrame:
    """Report histogram occupancy and branch-cut exposure per residue.

    A fixed support makes mutual information comparable across replicas, blocks and
    bootstrap samples, but it costs resolution when the dihedral movements are narrow:
    bins outside the sampled range sit empty. This quantifies that trade-off so the
    width of the support can be chosen from data rather than guessed.

    Args:
        df_all_residues (pd.DataFrame): Dihedral angle movements, one column per residue.

        num_bins (int): Number of histogram bins. Default is 35.

        hist_range (tuple | None): Fixed ``(low, high)`` support to evaluate. ``None``
            reports the data-derived binning instead.

        csv_path (str): Where the per-residue table is written.

    Returns:
        pd.DataFrame: One row per residue with ``residue``, ``occupied_bins``,
        ``occupied_fraction``, ``movement_min``, ``movement_max`` and, when the
        upstream calculation recorded it, ``branch_cut_crossings`` (the number of
        frames whose *unwrapped* difference exceeded 180 degrees).
    """
    crossings = df_all_residues.attrs.get("branch_cut_crossings", {})
    records = []
    for column in df_all_residues.columns:
        values = np.asarray(df_all_residues[column], dtype=float)
        values = values[np.isfinite(values)]
        counts, _ = np.histogram(values, bins=num_bins, range=hist_range)
        record = {
            "residue": column,
            "occupied_bins": int(np.count_nonzero(counts)),
            "occupied_fraction": float(np.count_nonzero(counts) / num_bins),
            "movement_min": float(values.min()) if values.size else np.nan,
            "movement_max": float(values.max()) if values.size else np.nan,
        }
        try:
            record["branch_cut_crossings"] = int(crossings[residue_id(column)])
        except (KeyError, ValueError):
            record["branch_cut_crossings"] = np.nan
        records.append(record)

    frame = pd.DataFrame(records)
    if csv_path:
        frame.to_csv(csv_path, index=False)

    occupied = frame["occupied_bins"].to_numpy()
    low, high = np.percentile(occupied, [5, 95]) if occupied.size else (np.nan, np.nan)
    sparse = int(np.count_nonzero(occupied < 10))
    print(
        f"Binning diagnostic ({num_bins} bins, support "
        f"{'data-derived' if hist_range is None else f'{hist_range[0]:g}..{hist_range[1]:g}'}): "
        f"occupied bins median {np.median(occupied):.0f}, 5th/95th pct {low:.0f}/{high:.0f}; "
        f"{sparse}/{len(occupied)} residues ({sparse / max(1, len(occupied)):.1%}) occupy <10 bins."
    )
    exposure = frame["branch_cut_crossings"]
    if exposure.notna().any():
        affected = int((exposure.fillna(0) > 0).sum())
        print(
            f"  branch-cut exposure: {affected}/{len(exposure)} residues "
            f"({affected / max(1, len(exposure)):.1%}) had at least one frame with "
            f"|dphi| > 180 before wrapping; {int(exposure.fillna(0).sum())} frames total."
        )
    if csv_path:
        print(f"  per-residue values written to {csv_path}")
    return frame
