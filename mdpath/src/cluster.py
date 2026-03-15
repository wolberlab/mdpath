"""Clustering --- :mod:`mdpath.src.cluster`
==============================================================================

This module contains the class `PatwayClustering` which calculates the overlap between pathways and clusters them based on the overlap.
Clusters are generated through hirarcical clustering using scipy. Optimal cluster size is evaluated using the silhouette score.

Classes
--------

:class:`PatwayClustering`
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
from tqdm import tqdm
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


class PatwayClustering:
    """Perform clustering of pathways based on the overlap of close residue pairs.

    Attributes:
        df (pd.DataFrame): DataFrame containing close residue pairs.

        pathways (list): List of pathways, where each pathway is a list of residue indices.

        num_processes (int): Number of processes to use for parallel computation.

        overlapp_df (pd.DataFrame): DataFrame containing the overlap between all pathway pairs.
    """

    def __init__(
        self, df_close_res: pd.DataFrame, pathways: list, num_processes: int
    ) -> None:
        self.df = df_close_res
        self.close_pairs_set = set(zip(df_close_res['Residue1'], df_close_res['Residue2'])) | \
                      set(zip(df_close_res['Residue2'], df_close_res['Residue1']))
        self.pathways = pathways
        self.num_processes = num_processes
        self.overlapp_df = self.calculate_overlap_parallel()

    def calculate_overlap_for_pathway(self, args: tuple) -> list:
        """Calculates the overlap between a pathway and all other pathways.

        Args:
            args (tuple): Argument wrapper conatining the pathway index, the pathway, all pathways and the dataframe with close residue pairs.

        Returns:
            result (list): List of dictionaries with the overlap between the given pathway and all other pathways.
        """
        i, path1 = args
        result = []
        for j in range(i + 1, len(self.pathways)):
            if i != j:
                path2 = self.pathways[j]
                overlap_counter = 0
                for res1 in path1:
                    for res2 in path2:
                        if (res1, res2) in self.close_pairs_set:
                            overlap_counter += 1
                result.append(
                    {"Pathway1": i, "Pathway2": j, "Overlap": overlap_counter}
                )
                result.append(
                    {"Pathway1": j, "Pathway2": i, "Overlap": overlap_counter}
                )
        return result

    def calculate_overlap_parallel(self) -> pd.DataFrame:
        """Parallelization wrapper for the calculate_overlap_for_pathway function.

        Returns:
            overlap_df (pd.DataFrame): Pandas dataframe with the overlap between all pathways and all other pathways.
        """
        args = [(i, path) for i, path in enumerate(self.pathways)]
        results = []
        with Pool(processes=self.num_processes) as pool:
            with tqdm(
                total=len(args),
                ascii=True,
                desc="\033[1mCalculating pathway residue overlap\033[0m",
            ) as pbar:
                for result in pool.imap_unordered(
                    self.calculate_overlap_for_pathway, args
                ):
                    results.extend(result)
                    pbar.update(1)

        overlap_df = pd.DataFrame(results, columns=["Pathway1", "Pathway2", "Overlap"])
        return overlap_df

    def pathways_cluster(
        self, n_top_clust: int = 0, save_path: str = "clustered_paths.png"
    ) -> dict:
        """Clustering of pathways based on the overlap between them.

        Args:
            n_top_clust (int, optional): Number of clusters to output. Defaults to all.

            save_path (str, optional): Save path for cluster dendogram figure. Defaults to "clustered_paths.png".

        Returns:
            clusters (dict): Dictionary with the clusters and their pathways.
        """
        overlap_matrix = self.overlapp_df.pivot(
            index="Pathway1", columns="Pathway2", values="Overlap"
        ).fillna(0)
        distance_matrix = 1 - overlap_matrix
        linkage_matrix = hierarchy.linkage(distance_matrix.values, method="complete")

        silhouette_scores = []
        for n_clusters in range(2, len(overlap_matrix)):
            cluster_labels = hierarchy.fcluster(
                linkage_matrix, n_clusters, criterion="maxclust"
            )
            silhouette_scores.append(silhouette_score(distance_matrix, cluster_labels))

        optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        print("Optimal number of clusters:", optimal_num_clusters)

        cluster_labels = hierarchy.fcluster(
            linkage_matrix, optimal_num_clusters, criterion="maxclust"
        )
        cluster_pathways = {
            cluster: [] for cluster in range(1, optimal_num_clusters + 1)
        }
        for i, label in enumerate(cluster_labels):
            cluster_pathways[label].append(overlap_matrix.index[i])

        silhouette_avg = silhouette_score(distance_matrix, cluster_labels)
        print("Silhouette Score:", silhouette_avg)
        plt.figure(figsize=(10, 7))
        hierarchy.dendrogram(
            linkage_matrix, labels=overlap_matrix.index, leaf_rotation=90
        )
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Pathways")
        plt.ylabel("Distance")
        plt.savefig(save_path)
        plt.close()

        sorted_clusters = sorted(
            cluster_pathways.items(), key=lambda x: len(x[1]), reverse=True
        )
        clusters = {}
        if n_top_clust == 0:
            for cluster, pathways in sorted_clusters:
                print(f"Cluster {cluster} (Size: {len(pathways)})")
                clusters[cluster] = pathways
        else:
            for cluster, pathways in sorted_clusters[:n_top_clust]:
                print(f"Cluster {cluster} (Size: {len(pathways)})")
                clusters[cluster] = pathways
        return clusters

    def pathway_clusters_dictionary(self, clusters: dict, sorted_paths: list) -> dict:
        """Generates a dictionary mapping cluster numbers to lists of pathways.

        Args:
            clusters (dict): A dictionary where keys are cluster numbers and values are lists of pathway IDs.

            sorted_paths (list): A list of pathways, where each pathway is a tuple and the first element is the pathway name.

        Returns:
            dict: A dictionary where keys are cluster numbers and values are lists of pathways corresponding to each cluster.
        """
        cluster_pathways_dict = {}
        for cluster_num, cluster_pathways in clusters.items():
            cluster_pathways_list = []
            for pathway_id in cluster_pathways:
                pathway = sorted_paths[pathway_id]
                cluster_pathways_list.append(pathway[0])
            cluster_pathways_dict[cluster_num] = cluster_pathways_list
        return cluster_pathways_dict
