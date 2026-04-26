"""Graph --- :mod:`mdpath.src.graph`
==============================================================================

This module contains the class `GraphBuilder` which generates a graph of residues within a certain distance of each other.
Graph edges are assigned weights based on mutual information differences.
Paths between distant residues are calculated based on the shortest path with the highest total weight.

Classes
--------

:class:`GraphBuilder`
"""

import heapq
import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial import cKDTree
from Bio import PDB
from typing import Tuple, List
from multiprocessing import Pool
from tqdm import tqdm
from mdpath.src.structure import StructureCalculations


def _max_weight_shortest_path(graph: nx.Graph, source: int, target: int) -> Tuple:
    """Shortest path between source and target with the maximum total edge weight
    among all shortest (fewest-hop) paths. Module-level so it can run in workers
    without pickling a GraphBuilder instance.
    """
    best = {source: (0, 0)}
    heap = [(0, 0, source, [source])]

    while heap:
        dist, neg_w, u, path = heapq.heappop(heap)
        acc_w = -neg_w

        if u == target:
            return path, acc_w

        prev_dist, prev_w = best.get(u, (float("inf"), -float("inf")))
        if dist > prev_dist or (dist == prev_dist and acc_w < prev_w):
            continue

        for v in graph.neighbors(u):
            edge_w = graph[u][v].get("weight", 0)
            new_dist = dist + 1
            new_w = acc_w + edge_w

            prev_v = best.get(v, (float("inf"), -float("inf")))
            if new_dist < prev_v[0] or (new_dist == prev_v[0] and new_w > prev_v[1]):
                best[v] = (new_dist, new_w)
                heapq.heappush(heap, (new_dist, -new_w, v, path + [v]))

    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")


_WORKER_GRAPH: "nx.Graph | None" = None


def _init_path_worker(graph: nx.Graph) -> None:
    """Pool initializer: stash the residue graph in a module global so workers
    reuse it across tasks instead of receiving it as part of every pickled task.
    """
    global _WORKER_GRAPH
    _WORKER_GRAPH = graph


def _worker_calc_path(residue_pair: tuple):
    """Pool task: compute the max-weight shortest path for a single residue pair
    using the worker-local graph set up by _init_path_worker.
    """
    res1, res2 = residue_pair
    try:
        return _max_weight_shortest_path(_WORKER_GRAPH, res1, res2)
    except nx.NetworkXNoPath:
        return None


class GraphBuilder:
    """Build and analyze residue interaction graphs based on residue distances and mutual information between residue pais.

    Attributes:
        pdb (str): Path to the PDB file.

        end (int): The last residue number to consider in the graph.

        mi_diff_df (pd.DataFrame): DataFrame containing mutual information differences between residue pairs.

        dist (int): Cutoff distance for graph edges in Angstroms.

        graph (nx.Graph): The constructed residue interaction graph.
    """

    def __init__(
        self, pdb: str, last_residue: int, mi_diff_df: pd.DataFrame, graphdist: int
    ) -> None:
        self.pdb = pdb
        self.end = last_residue
        self.mi_diff_df = mi_diff_df
        self.dist = graphdist
        self.graph = self.graph_builder()

    def graph_skeleton(self) -> nx.Graph:
        """Generates a graph of residues with edges for residues within in a given distance of each other.

        Returns:
            residue_graph (nx.Graph): Graph of residues within a certain distance of each other.
        """
        residue_graph = nx.Graph()
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("pdb_structure", self.pdb)
        heavy_atoms = {"C", "N", "O", "S"}
        residues = [
            res for res in structure.get_residues() if PDB.Polypeptide.is_aa(res)
        ]

        coords = []
        res_ids = []
        for res in residues:
            rid = res.get_id()[1]
            if rid <= self.end:
                for atom in res:
                    if atom.element in heavy_atoms:
                        coords.append(atom.coord)
                        res_ids.append(rid)

        if not coords:
            return residue_graph

        coords = np.array(coords)
        res_ids = np.array(res_ids)
        tree = cKDTree(coords)
        atom_pairs = tree.query_pairs(r=self.dist)

        for i, j in atom_pairs:
            if res_ids[i] != res_ids[j]:
                residue_graph.add_edge(int(res_ids[i]), int(res_ids[j]), weight=0)

        return residue_graph

    def graph_assign_weights(self, residue_graph: nx.Graph) -> nx.Graph:
        """Assignes edge weights to the graph based on mutual information between the residue pair.

        Args:
            residue_graph (nx.Graph): Base residue graph (graph skeleton).

        Returns:
            residue_graph (nx.Graph): Residue graph with edge weights assigned.
        """
        weight_lookup = {}
        for _, row in self.mi_diff_df.iterrows():
            pair = tuple(row["Residue Pair"])
            weight_lookup[pair] = row["MI Difference"]

        for edge in residue_graph.edges():
            u, v = edge
            pair = ("Res " + str(u), "Res " + str(v))
            if pair in weight_lookup:
                residue_graph.edges[edge]["weight"] = weight_lookup[pair]
        return residue_graph

    def graph_builder(self) -> nx.Graph:
        """Wrapper function to build the residue graph.

        Returns:
            residue_graph (nx.Graph): Full residue graph with edge weights assigned.
        """
        graph = self.graph_skeleton()
        graph = self.graph_assign_weights(graph)
        return graph

    def max_weight_shortest_path(self, source: int, target: int) -> Tuple:
        """Finds the shortest path between 2 nodes with the highest total weight among all shortest paths.

        Args:
            source (int): Starting node.

            target (int): Target node.

        Returns:
            best_path (List): List of nodes in the shortest path with the highest weight.

            total_weight (float): Total weight of the shortest path.
        """
        return _max_weight_shortest_path(self.graph, source, target)

    def collect_path_total_weights(self, df_distant_residues: pd.DataFrame) -> list:
        """Wrapper function to collect the shortest path and total weight between distant residues.

        Args:
            residue_graph (nx.Graph): Residue graph.

            df_distant_residues (pd.DataFrame): Panda dataframe with distant residues.

        Returns:
            path_total_weights (list): List of tuples with the shortest path and total weight between distant residues.
        """
        path_total_weights = []
        for index, row in df_distant_residues.iterrows():
            try:
                shortest_path, total_weight = self.max_weight_shortest_path(
                    row["Residue1"], row["Residue2"]
                )
                path_total_weights.append((shortest_path, total_weight))
            except nx.NetworkXNoPath:
                continue
        return path_total_weights

    def calc_path_weight(self, residue_pair: tuple) -> tuple:
        """Calculates the shortest path and total weight for a single residue pair.

        Args:
            residue_pair (tuple): Tuple of (Residue1, Residue2).

        Returns:
            tuple | None: (shortest_path, total_weight) or None if no path exists.
        """
        res1, res2 = residue_pair
        try:
            shortest_path, total_weight = self.max_weight_shortest_path(res1, res2)
            return (shortest_path, total_weight)
        except nx.NetworkXNoPath:
            return None

    def collect_path_total_weights_parallel(
        self, df_distant_residues: pd.DataFrame, num_parallel_processes: int
    ) -> list:
        """Parallel wrapper to collect the shortest path and total weight between distant residues.

        Args:
            df_distant_residues (pd.DataFrame): DataFrame with distant residues (columns: Residue1, Residue2).

            num_parallel_processes (int): Number of parallel processes.

        Returns:
            path_total_weights (list): List of tuples with the shortest path and total weight between distant residues.
        """
        residue_pairs = [
            (row["Residue1"], row["Residue2"])
            for _, row in df_distant_residues.iterrows()
        ]
        path_total_weights = []
        # Change
        # Pickle the graph once per worker via initializer/initargs instead of
        # once per task (which is what happens when a bound method is dispatched).
        chunksize = max(1, len(residue_pairs) // (num_parallel_processes * 4)) if residue_pairs else 1
        with Pool(
            processes=num_parallel_processes,
            initializer=_init_path_worker,
            initargs=(self.graph,),
        ) as pool:
            with tqdm(
                total=len(residue_pairs),
                ascii=True,
                desc="\033[1mCalculating path total weights\033[0m",
            ) as pbar:
                results = pool.imap_unordered(
                    _worker_calc_path, residue_pairs, chunksize=chunksize
                )
                for result in results:
                    if result is not None:
                        path_total_weights.append(result)
                    pbar.update(1)
        return path_total_weights
