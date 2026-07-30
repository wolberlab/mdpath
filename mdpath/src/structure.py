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
from scipy.spatial import cKDTree
import logging


# Bio.PDB.Polypeptide.is_aa() only knows PDB chemical-component names, so it
# rejects the protonation-state names that MD force fields write (CHARMM HSD/HSP,
# AMBER HID/HIE/CYX/ASH/...). Residues it rejects are silently dropped from the
# graph, which loses real amino acids without any warning. It also happens to
# accept HSE, but as homoserine rather than epsilon-protonated histidine.
FORCEFIELD_RESNAME_ALIASES = {
    # CHARMM
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    # AMBER
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "CYX": "CYS",
    "CYM": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
    # GROMOS / misc terminal variants
    "CYS2": "CYS",
    "HISA": "HIS",
    "HISB": "HIS",
    "HISH": "HIS",
}


def is_amino_acid(residue) -> bool:
    """Checks whether a residue is an amino acid, including force field variants.

    Args:
        residue: A ``Bio.PDB`` residue, or a residue name.

    Returns:
        bool: True for standard amino acids and for force field protonation
        variants such as ``HSD``, ``HID`` or ``CYX``.
    """
    name = residue if isinstance(residue, str) else residue.get_resname()
    name = name.strip().upper()
    return PDB.Polypeptide.is_aa(FORCEFIELD_RESNAME_ALIASES.get(name, name))


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
            if is_amino_acid(res):
                res_num = res.id[1]
                if res_num < first_res_num:
                    first_res_num = res_num
                if res_num > last_res_num:
                    last_res_num = res_num
        return int(first_res_num), int(last_res_num)

    def _build_kdtree(self, dist: float):
        """Builds a KDTree from heavy atoms and returns close residue pairs.

        Args:
            dist (float): Distance cutoff.

        Returns:
            close_res_pairs (set): Set of (min_id, max_id) tuples for close residue pairs.
            all_unique_res (list): Sorted list of all unique residue IDs.
        """
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("pdb_structure", self.pdb)
        heavy_atoms = {"C", "N", "O", "S"}
        residues = [res for res in structure.get_residues() if is_amino_acid(res)]

        coords = []
        res_ids = []
        for res in residues:
            rid = res.get_id()[1]
            if rid <= self.last_res_num:
                for atom in res:
                    if atom.element in heavy_atoms:
                        coords.append(atom.coord)
                        res_ids.append(rid)

        if not coords:
            return set(), []

        coords = np.array(coords)
        res_ids = np.array(res_ids)
        tree = cKDTree(coords)
        atom_pairs = tree.query_pairs(r=dist)

        close_res_pairs = set()
        for i, j in atom_pairs:
            r1, r2 = int(res_ids[i]), int(res_ids[j])
            if r1 != r2:
                close_res_pairs.add((min(r1, r2), max(r1, r2)))

        all_unique_res = sorted(set(res_ids.tolist()))
        return close_res_pairs, all_unique_res

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

        close_res_pairs, all_unique_res = self._build_kdtree(dist)

        if not all_unique_res:
            return pd.DataFrame(columns=["Residue1", "Residue2"])

        if mode == "close":
            residue_pairs = sorted(close_res_pairs)
        else:
            all_res_pairs = set(
                (min(r1, r2), max(r1, r2)) for r1, r2 in combinations(all_unique_res, 2)
            )
            residue_pairs = sorted(all_res_pairs - close_res_pairs)

        return pd.DataFrame(residue_pairs, columns=["Residue1", "Residue2"])

    def calculate_close_and_far(self, dist: float):
        """Calculates both close and far residue pairs in a single KDTree pass.

        Args:
            dist (float): Distance cutoff for residue pairs.

        Returns:
            df_close (pd.DataFrame): Close residue pairs.
            df_far (pd.DataFrame): Far residue pairs.
        """
        close_res_pairs, all_unique_res = self._build_kdtree(dist)

        if not all_unique_res:
            empty = pd.DataFrame(columns=["Residue1", "Residue2"])
            return empty, empty.copy()

        all_res_pairs = set(
            (min(r1, r2), max(r1, r2)) for r1, r2 in combinations(all_unique_res, 2)
        )
        far_res_pairs = all_res_pairs - close_res_pairs

        df_close = pd.DataFrame(
            sorted(close_res_pairs), columns=["Residue1", "Residue2"]
        )
        df_far = pd.DataFrame(sorted(far_res_pairs), columns=["Residue1", "Residue2"])
        return df_close, df_far

    def calculate_dihedral_movements_multi_traj(
        self,
        traj_files: list,
        topology: str,
        num_parallel_processes: int,
        return_per_replica: bool = False,
    ):
        """Process multiple independent trajectories and pool dihedral movement results.

        Each trajectory is processed independently to avoid artificial jumps
        at trajectory boundaries when computing dihedral angle differences.

        Args:
            traj_files (list): List of trajectory file paths.
            topology (str): Path to the topology file.
            num_parallel_processes (int): Number of parallel processes for dihedral calculation.
            return_per_replica (bool, optional): If True, also return the list of per-replica
                DataFrames aligned to the common residue columns. Default False preserves the
                original single-return behavior.

        Returns:
            pd.DataFrame: Concatenated DataFrame of dihedral angle movements from all trajectories.
            If ``return_per_replica`` is True, returns a tuple ``(concatenated_df, per_replica_dfs)``
            where ``per_replica_dfs`` is a list of DataFrames (one per replica) restricted to the
            same common columns as the concatenated output.
        """
        all_dfs = []
        for i, traj_file in enumerate(traj_files):
            print(
                f"\033[1mProcessing trajectory {i + 1}/{len(traj_files)}: {traj_file}\033[0m"
            )
            traj = mda.Universe(topology, traj_file)
            dihedral_calc = DihedralAngles(
                traj, self.first_res_num, self.last_res_num, self.num_residues
            )
            df = dihedral_calc.calculate_dihedral_movement_parallel(
                num_parallel_processes
            )
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            empty = pd.DataFrame()
            if return_per_replica:
                return empty, []
            return empty

        # Only keep residues present in ALL replicas
        common_cols = set(all_dfs[0].columns)
        for df in all_dfs[1:]:
            common_cols &= set(df.columns)
        common_cols = sorted(common_cols)

        dropped = set(all_dfs[0].columns)
        for df in all_dfs:
            dropped |= set(df.columns)
        n_dropped = len(dropped) - len(common_cols)
        if n_dropped > 0:
            print(
                f"\033[1mWarning: {n_dropped} residue(s) dropped due to inconsistent "
                f"dihedral data across replicas.\033[0m"
            )

        aligned = []
        for df in all_dfs:
            attrs = df.attrs.copy()
            df = df[common_cols].copy()
            df.attrs = attrs
            aligned.append(df)
        all_dfs = aligned
        concatenated = pd.concat(all_dfs, axis=0, ignore_index=True)
        if return_per_replica:
            return concatenated, all_dfs
        return concatenated


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
        # Map resid -> position in traj.residues. Never do this by arithmetic:
        # numbering gaps (missing loops) or repeated resids silently shift every
        # residue after the discontinuity onto the wrong dihedral.
        self._res_pos = {}
        duplicates = []
        for pos, res in enumerate(traj.residues):
            # ligands, ions and waters routinely reuse the protein's numbering;
            # only amino acids become graph nodes, so only they need to be unique
            if not is_amino_acid(res.resname):
                continue
            resid = int(res.resid)
            if resid in self._res_pos:
                duplicates.append(resid)
            else:
                self._res_pos[resid] = pos
        if duplicates:
            raise ValueError(
                f"{len(duplicates)} amino acid(s) in the topology share a residue "
                f"number with another (e.g. {sorted(set(duplicates))[:10]}). MDPath "
                "keys everything by residue number, so this is ambiguous. Usual "
                "cause: PDB insertion codes (205A, 205B, ...) collapsed onto the "
                "base number because the topology came from a format without an "
                "insertion-code field (PSF/GRO). Renumber the residues sequentially "
                "before running."
            )

    def _calc_dihedral_movement_with_crossings(self, res_id: int) -> tuple:
        """Calculate wrapped dihedral movement and count branch-cut crossings.

        Args:
            res_id (int): Residue number.

        Returns:
            tuple[int, np.ndarray, int] | None: ``(residue_id, movement, crossings)``
            where ``crossings`` is the number of frames whose *unwrapped* difference
            exceeded 180 degrees, i.e. the residue's exposure to the periodic branch
            cut. ``None`` if the dihedral could not be calculated.
        """
        try:
            res = self.traj.residues[self._res_pos[res_id]]
            ags = [res.phi_selection()]
            if not all(ags):
                return None
            R = Dihedral(ags).run()
            dihedrals = R.results.angles
            raw = np.diff(dihedrals, axis=0)
            # phi is periodic on [-180, 180); an unwrapped difference turns a
            # small step across the branch cut into a spurious ~360 deg jump,
            # which then dominates the histogram support for that residue.
            dihedral_angle_movement = (raw + 180.0) % 360.0 - 180.0
            crossings = int(np.count_nonzero(np.abs(raw) > 180.0))
            return res_id, dihedral_angle_movement, crossings
        except (TypeError, AttributeError, IndexError) as e:
            logging.debug(
                f"Failed to calculate dihedral for residue {res_id}: {str(e)}"
            )
            return None

    def calc_dihedral_angle_movement(self, res_id: int) -> tuple:
        """Calculates dihedral angle movement for a residue over the course of the MD trajectory.

        The difference is wrapped onto ``[-180, 180)`` because phi is periodic; see
        :meth:`_calc_dihedral_movement_with_crossings`.

        Args:
            res_id (int): Residue number.

        Returns:
            tuple[int, np.ndarray] | None: Tuple of (residue_id, dihedral_angles) if successful, None if failed.
        """
        result = self._calc_dihedral_movement_with_crossings(res_id)
        return None if result is None else result[:2]

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
        collected = []
        branch_cut_crossings = {}

        try:
            with Pool(processes=num_parallel_processes) as pool:
                # iterate the resids that actually exist, not first..last: a
                # numbering gap would otherwise queue residues that aren't there
                res_ids = sorted(self._res_pos)
                with tqdm(
                    total=len(res_ids),
                    ascii=True,
                    desc="\033[1mProcessing residue dihedral movements\033[0m",
                ) as pbar:
                    results = pool.imap_unordered(
                        self._calc_dihedral_movement_with_crossings,
                        res_ids,
                    )

                    for result in results:
                        if result is None:
                            pbar.update(1)
                            continue

                        res_id, dihedral_data, crossings = result
                        branch_cut_crossings[res_id] = crossings
                        try:
                            collected.append(
                                pd.DataFrame(dihedral_data, columns=[f"Res {res_id}"])
                            )
                        except Exception as e:
                            logging.error(
                                f"\033[1mError processing residue {res_id}: {e}\033[0m"
                            )
                        finally:
                            pbar.update(1)

        except Exception as e:
            logging.error(f"Parallel processing failed: {str(e)}")

        if collected:
            result = pd.concat(collected, axis=1)
            # branch-cut exposure per residue, for the binning diagnostic
            result.attrs["branch_cut_crossings"] = branch_cut_crossings
            return result
        return pd.DataFrame()
