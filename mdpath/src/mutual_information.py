"""Mutual Information Calculation --- :mod:`mdpath.src.mutual_information`
===============================================================================

This module contains the class `NMICalculator` which calculates the Normalized Mutual Information (NMI)
for all residue pairs in a given dataset based on the dihedral angle movements over the course of the analysed MD trajectory.


Classes
--------

:class:`NMICalculator`
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
    ) -> None:
        self.df_all_residues = df_all_residues
        self.num_bins = num_bins
        self.invert = invert
        self.pairs = pairs
        self.show_progress = show_progress
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
            hist, _ = np.histogram(self.df_all_residues[col], bins=self.num_bins)
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
