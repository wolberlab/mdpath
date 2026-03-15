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
from sklearn.metrics import mutual_info_score
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from scipy.special import digamma


class NMICalculator:
    """Calculate Normalized Mutual Information (NMI) between dihedral angle movements of residue pairs.

    Attributes:
        df_all_residues (pd.DataFrame): DataFrame containing all residues.

        num_bins (int): Number of bins to use for histogram calculations. Default is 35.


        nmi_df (pd.DataFrame): DataFrame containing the mutual information differences. Is calculated using either GMM or histogram method.

        entropy_df (pd.DataFrame): Pandas dataframe with residue and entropy values. Is calculated using either GMM or histogram method.
    """

    def __init__(
        self,
        df_all_residues: pd.DataFrame,
        num_bins: int = 35,
        invert=False,
    ) -> None:
        self.df_all_residues = df_all_residues
        self.num_bins = num_bins
        self.invert = invert
        self.nmi_df, self.entropy_df = self.NMI_calcs()

    def NMI_calcs(self):
        """Extended Normalized Mutual Information and Entropy calculation."""
        entropys = {}
        normalized_mutual_info = {}
        total_iterations = len(self.df_all_residues.columns) ** 2
        with tqdm(
            total=total_iterations,
            desc="\033[1mCalculating Normalized Mutual Information\033[0m",
        ) as progress_bar:
            for col1 in self.df_all_residues.columns:
                for col2 in self.df_all_residues.columns:
                    if col1 != col2:
                        hist_col1, _ = np.histogram(
                            self.df_all_residues[col1], bins=self.num_bins
                        )
                        hist_col2, _ = np.histogram(
                            self.df_all_residues[col2], bins=self.num_bins
                        )
                        hist_joint, _, _ = np.histogram2d(
                            self.df_all_residues[col1],
                            self.df_all_residues[col2],
                            bins=self.num_bins,
                        )
                        mi = mutual_info_score(
                            hist_col1, hist_col2, contingency=hist_joint
                        )
                        entropy_col1 = entropy(hist_col1)
                        entropy_col2 = entropy(hist_col2)
                        entropys[col1] = entropy_col1
                        entropys[col2] = entropy_col2
                        nmi = mi / np.sqrt(entropy_col1 * entropy_col2)
                        normalized_mutual_info[(col1, col2)] = nmi
                        progress_bar.update(1)

        entropy_df = pd.DataFrame(entropys.items(), columns=["Residue", "Entropy"])
        nmi_df = pd.DataFrame(
            normalized_mutual_info.items(), columns=["Residue Pair", "MI Difference"]
        )
        if self.invert:
            max_nmi_diff = nmi_df["MI Difference"].max()
            nmi_df["MI Difference"] = max_nmi_diff - nmi_df["MI Difference"]
        return nmi_df, entropy_df
