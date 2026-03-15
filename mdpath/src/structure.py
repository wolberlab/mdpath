"""Structure Calculations --- :mod:`mdpath.src.structure`
==============================================================================

This module contains the class `StructureCalculations` which handels all structure related calculations based on the initial PDB file
and the  class `DihedralAngles` which calculates the dihedral angle movements over the course of the given trajectory.

Classes
---------

:class:`StructureCalculations`
:class:`DihedralAngles`
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from multiprocessing import Pool
from Bio import PDB
from itertools import combinations
import logging


class StructureCalculations:
    """Calculate residue surroundings and distances between residues in a PDB structure.

    Attributes:
        pdb (str): Path to the PDB file.

        first_res_num (int): First residue number in the PDB file.

        last_res_num (int): Last residue number in the PDB file.

        num_residues (int): Total number of residues in the PDB file.
    """

    def __init__(self, pdb: str) -> None:
        self.pdb = pdb
        self.first_res_num, self.last_res_num = self.res_num_from_pdb()
        self.num_residues = self.last_res_num - self.first_res_num + 1

    def res_num_from_pdb(self) -> tuple:
        """Gets first and last residue number from a PDB file.

        Returns:
            first_res_num (int): First residue number.

            last_res_num (int): Last residue number.
        """
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", self.pdb)
        first_res_num = float("inf")
        last_res_num = float("-inf")
        for res in structure.get_residues():
            if PDB.Polypeptide.is_aa(res):
                res_num = res.id[1]
                if res_num < first_res_num:
                    first_res_num = res_num
                if res_num > last_res_num:
                    last_res_num = res_num
        return int(first_res_num), int(last_res_num)

    def calculate_distance(self, atom1: tuple, atom2: tuple) -> float:
        """Calculates the distance between two atoms.

        Args:
            atom1 (tuple): Coordinates of the first atom.

            atom2 (tuple): Coordinates of the second atom.

        Returns:
            distance (float): Normalized distance between the two atoms.
        """
        distance_vector = [
            atom1[i] - atom2[i] for i in range(min(len(atom1), len(atom2)))
        ]
        distance = np.linalg.norm(distance_vector)
        return distance

    def calculate_residue_suroundings(self, dist: float, mode: str) -> pd.DataFrame:
        """Calculates residues that are either close to or far away from each other in a PDB structure.

        Args:
            dist (float): Distance cutoff for residue pairs.

            mode (str): 'close' to calculate close residues, 'far' to calculate faraway residues.

        Returns:
            pd.DataFrame: Pandas dataframe with residue pairs and their distance.
        """
        if mode not in ["close", "far"]:
            raise ValueError("Mode must be either 'close' or 'far'.")

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("pdb_structure", self.pdb)
        heavy_atoms = ["C", "N", "O", "S"]
        residue_pairs = []
        residues = [
            res for res in structure.get_residues() if PDB.Polypeptide.is_aa(res)
        ]

        for res1, res2 in tqdm(
            combinations(residues, 2),
            desc=f"\033[1mCalculating {mode} residue surroundings\033[0m",
            total=len(residues) * (len(residues) - 1) // 2,
        ):
            res1_id = res1.get_id()[1]
            res2_id = res2.get_id()[1]
            if res1_id <= self.last_res_num and res2_id <= self.last_res_num:
                condition_met = False if mode == "close" else True
                for atom1 in res1:
                    if atom1.element in heavy_atoms:
                        for atom2 in res2:
                            if atom2.element in heavy_atoms:
                                distance = self.calculate_distance(
                                    atom1.coord, atom2.coord
                                )
                                if (mode == "close" and distance <= dist) or (
                                    mode == "far" and distance > dist
                                ):
                                    condition_met = True
                                    break
                        if condition_met:
                            break
                if condition_met:
                    residue_pairs.append((res1_id, res2_id))

        return pd.DataFrame(residue_pairs, columns=["Residue1", "Residue2"])


class DihedralAngles:
    """Calculate dihedral angle movements for residues in a molecular dynamics (MD) trajectory.

    Attributes:
        traj (mda.Universe): MDAnalysis Universe object containing the trajectory.

        first_res_num (int): The first residue number in the trajectory.

        last_res_num (int): The last residue number in the trajectory.

        num_residues (int): The total number of residues in the trajectory.

    """

    def __init__(
        self,
        traj: mda.Universe,
        first_res_num: int,
        last_res_num: int,
        num_residues: int,
    ) -> None:
        self.traj = traj
        self.first_res_num = first_res_num
        self.last_res_num = last_res_num
        self.num_residues = num_residues

    def calc_dihedral_angle_movement(self, res_id: int) -> tuple:
        """Calculates dihedral angle movement for a residue over the course of the MD trajectory.

        Args:
            res_id (int): Residue number.

        Returns:
            tuple[int, np.ndarray] | None: Tuple of (residue_id, dihedral_angles) if successful, None if failed.
        """
        try:
            res = self.traj.residues[res_id]
            ags = [res.phi_selection()]
            if not all(ags):  # Check if any selections are None
                return None
            R = Dihedral(ags).run()
            dihedrals = R.results.angles
            dihedral_angle_movement = np.diff(dihedrals, axis=0)
            return res_id, dihedral_angle_movement
        except (TypeError, AttributeError, IndexError) as e:
            logging.debug(
                f"Failed to calculate dihedral for residue {res_id}: {str(e)}"
            )
            return None

    def calculate_dihedral_movement_parallel(
        self,
        num_parallel_processes: int,
    ) -> pd.DataFrame:
        """Parallel calculation of dihedral angle movement for all residues in the trajectory.

        Args:
            num_parallel_processes (int): Amount of parallel processes.

        Returns:
            pd.DataFrame: DataFrame with all residue dihedral angle movements.
        """
        df_all_residues = pd.DataFrame()

        try:
            with Pool(processes=num_parallel_processes) as pool:
                with tqdm(
                    total=self.num_residues,
                    ascii=True,
                    desc="\033[1mProcessing residue dihedral movements\033[0m",
                ) as pbar:
                    results = pool.imap_unordered(
                        self.calc_dihedral_angle_movement,
                        range(self.first_res_num, self.last_res_num + 1),
                    )

                    for result in results:
                        if result is None:
                            pbar.update(1)
                            continue

                        res_id, dihedral_data = result
                        try:
                            df_residue = pd.DataFrame(
                                dihedral_data, columns=[f"Res {res_id}"]
                            )
                            df_all_residues = pd.concat(
                                [df_all_residues, df_residue], axis=1
                            )
                        except Exception as e:
                            logging.error(
                                f"\033[1mError processing residue {res_id}: {e}\033[0m"
                            )
                        finally:
                            pbar.update(1)

        except Exception as e:
            logging.error(f"Parallel processing failed: {str(e)}")

        return df_all_residues
