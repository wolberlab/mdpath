"""MDPath --- MD signal transduction calculation and visualization --- :mod:`mdpath.mdapth`
====================================================================

MDPath is a Python package for calculating signal transduction paths in molecular dynamics (MD) simulations.
The package uses mutual information to identify connections between residue movements.
Using a graph shortest paths with the highest mutual information are calculated.
Paths are then clustered based on the overlap between them to identify a continuous network throught the analysed protein.
The package also includes functionalitys for the visualization of results.

Release under the MIT License. See LICENSE for details.

This is the main script of MDPath. It is used to run MDPath from the command line.
MDPath can be called from the comadline using 'mdapth' after instalation
Use the -h flag to see the available options.
"""

import os
import argparse
import pandas as pd
import MDAnalysis as mda
import json
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import pickle

from mdpath.src.structure import StructureCalculations, DihedralAngles
from mdpath.src.mutual_information import NMICalculator
from mdpath.src.graph import GraphBuilder
from mdpath.src.cluster import PatwayClustering
from mdpath.src.visualization import MDPathVisualize
from mdpath.src.bootstrap import BootstrapAnalysis
from mdpath.src.confidence import EdgeConfidenceCalculator


def main():
    """Main function for running MDPath from the command line.
    It can be called using 'mdpath' after installation.

    Command-line inputs:
        -top: Topology file of your MD simulation

        -traj: Trajectory file of your MD simulation

        -cpu: Amount of cores used in multiprocessing (default: half of available cores)

        -lig: Protein ligand interacting residues (default: False)

        -bs: How often bootstrapping should be performed (default: False)

        -fardist: Default distance for faraway residues (default: 12.0)

        -closedist: Default distance for close residues (default: 12.0)

        -graphdist: Default distance for residues making up the graph (default: 5.0)

        -numpath: Default number of top paths considered for clustering (default: 500)
    """
    parser = argparse.ArgumentParser(
        prog="mdpath",
        description="Calculate signal transduction paths in your MD trajectories",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-top",
        dest="topology",
        help="Topology file of your MD simulation",
        required=False,
    )
    parser.add_argument(
        "-traj",
        dest="trajectory",
        help="Trajectory file(s) of your MD simulation. Supports multiple replicas e.g. -traj traj1.dcd traj2.dcd",
        required=False,
        nargs="+",
    )
    parser.add_argument(
        "-cpu",
        dest="num_parallel_processes",
        help="Amount of cores used in multiprocessing",
        default=(os.cpu_count() // 2),
    )
    parser.add_argument(
        "-lig",
        dest="lig_interaction",
        help="Protein ligand interacting residues",
        default=False,
        nargs="+",
    )
    parser.add_argument(
        "-bs",
        dest="bootstrap",
        help="How often bootstrapping should be performed.",
        default=False,
    )
    parser.add_argument(
        "-spline",
        dest="spline",
        help="Create STL files for spline visualization.",
        default=True,
    )
    # TODO maybe move settingsflags to a conffile that can be changed
    # Settings Flags
    parser.add_argument(
        "-fardist",
        dest="fardist",
        help="Default distance for faraway residues.",
        required=False,
        default=12.0,
    )
    parser.add_argument(
        "-closedist",
        dest="closedist",
        help="Default distance for close residues.",
        required=False,
        default=12.0,
    )
    parser.add_argument(
        "-graphdist",
        dest="graphdist",
        help="Default distance for residues making up the graph.",
        required=False,
        default=5.0,
    )
    parser.add_argument(
        "-numpath",
        dest="numpath",
        help="Default number of top paths considered for clustering.",
        required=False,
        default=500,
    )

    parser.add_argument(
        "-chain",
        dest="chain",
        help="Chain of the protein to be analyzed in the topology file. CAUTION: only one chain can be selected for analysis.",
        required=False,
        default=False,
    )

    parser.add_argument(
        "-segid",
        dest="segid",
        help="Segment ID of the protein to be analyzed (e.g. PROA). For CHARMM topologies that use segid instead of chainID. CAUTION: only one segment can be selected for analysis.",
        required=False,
        default=False,
    )

    parser.add_argument(
        "-invert",
        dest="invert",
        help="Inverts NMI bei subtrackting each NMI from max NMI. Can be used to find Paths, that are the least correlated",
        required=False,
        default=False,
    )

    parser.add_argument(
        "-confidence",
        dest="confidence",
        help="Compute per-edge confidence across replicas. Requires multiple -traj files. "
        "Produces edge_confidence.csv (all graph edges) and top_paths_edge_confidence.csv "
        "(consecutive transitions along the top pathways).",
        required=False,
        default=False,
    )

    parser.add_argument(
        "-water",
        dest="water",
        help="Allows for the tracking of stable waters medigating allosteric communication. Only include if the trajectory includes the water model.",
        required=False,
        default=False,
    )

    args = parser.parse_args()
    if not args.topology or not args.trajectory:
        print("Both trajectory and topology files are required!")
        exit()

    if args.chain and args.segid:
        print("Error: -chain and -segid are mutually exclusive. Please use only one.")
        exit()

    num_parallel_processes = int(args.num_parallel_processes)
    topology = args.topology
    trajectories = args.trajectory  # list due to nargs="+"
    traj = mda.Universe(topology, trajectories[0])
    lig_interaction = args.lig_interaction
    bootstrap = args.bootstrap
    fardist = float(args.fardist)
    closedist = float(args.closedist)
    graphdist = float(args.graphdist)
    numpath = int(args.numpath)
    invert = bool(args.invert)
    spline = bool(args.spline)
    water = float(args.water)
    confidence = bool(args.confidence)
    if confidence and len(trajectories) < 2:
        print(
            "Error: -confidence requires multiple replica trajectories "
            "(-traj traj1 traj2 ...)."
        )
        exit()

    # Chain selection
    if args.chain:
        chain = str(args.chain)
        chain_atoms = traj.select_atoms(f"chainID {chain}")
        if len(chain_atoms) == 0:
            raise ValueError(f"No atoms found for chain {chain}")
        chain_universe = mda.Merge(chain_atoms)
        # Write new topology
        chain_universe.atoms.write("selected_chain.pdb")

        # Write trajectory for each replica
        new_trajectories = []
        for i, traj_file in enumerate(trajectories):
            u = mda.Universe(topology, traj_file)
            chain_sel = u.select_atoms(f"chainID {chain}")
            out_name = f"selected_chain_{i}.dcd"
            with mda.Writer(out_name, chain_sel.n_atoms) as W:
                for ts in u.trajectory:
                    chain_universe.atoms.positions = chain_sel.positions
                    W.write(chain_universe.atoms)
            new_trajectories.append(out_name)

        topology = "selected_chain.pdb"
        trajectories = new_trajectories
        traj = mda.Universe(topology, trajectories[0])
        print(
            f"Chain {chain} selected for {len(trajectories)} trajectory file(s) and will now be analyzed."
        )

    # Segment ID selection (CHARMM compatibility)
    if args.segid:
        segid = str(args.segid)
        segid_atoms = traj.select_atoms(f"segid {segid}")
        if len(segid_atoms) == 0:
            raise ValueError(f"No atoms found for segid {segid}")
        segid_universe = mda.Merge(segid_atoms)
        segid_universe.atoms.write("selected_segid.pdb")

        new_trajectories = []
        for i, traj_file in enumerate(trajectories):
            u = mda.Universe(topology, traj_file)
            segid_sel = u.select_atoms(f"segid {segid}")
            out_name = f"selected_segid_{i}.dcd"
            with mda.Writer(out_name, segid_sel.n_atoms) as W:
                for ts in u.trajectory:
                    segid_universe.atoms.positions = segid_sel.positions
                    W.write(segid_universe.atoms)
            new_trajectories.append(out_name)

        topology = "selected_segid.pdb"
        trajectories = new_trajectories
        traj = mda.Universe(topology, trajectories[0])
        print(
            f"Segment {segid} selected for {len(trajectories)} trajectory file(s) and will now be analyzed."
        )

    # Write first frame PDB after all selections
    if os.path.exists("first_frame.pdb"):
        os.remove("first_frame.pdb")
    with mda.Writer("first_frame.pdb", multiframe=False) as pdb:
        traj.trajectory[0]
        pdb.write(traj.atoms)

    topology = "first_frame.pdb"
    structure_calc = StructureCalculations(topology)

    # Single KDTree pass when fardist == closedist (the default)
    if fardist == closedist:
        df_close_res, df_distant_residues = structure_calc.calculate_close_and_far(
            fardist
        )
    else:
        df_distant_residues = structure_calc.calculate_residue_suroundings(
            fardist, "far"
        )
        df_close_res = structure_calc.calculate_residue_suroundings(closedist, "close")

    per_replica_dfs = None
    if len(trajectories) == 1:
        dihedral_calc = DihedralAngles(
            traj,
            structure_calc.first_res_num,
            structure_calc.last_res_num,
            structure_calc.num_residues,
        )
        df_all_residues = dihedral_calc.calculate_dihedral_movement_parallel(
            num_parallel_processes
        )
    else:
        if confidence:
            df_all_residues, per_replica_dfs = (
                structure_calc.calculate_dihedral_movements_multi_traj(
                    trajectories,
                    topology,
                    num_parallel_processes,
                    return_per_replica=True,
                )
            )
        else:
            df_all_residues = structure_calc.calculate_dihedral_movements_multi_traj(
                trajectories, topology, num_parallel_processes
            )
    print("\033[1mTrajectory is processed and ready for analysis.\033[0m")

    # Calculate the mutual information and build the graph
    nmi_calc = NMICalculator(df_all_residues, invert=invert)
    nmi_calc.entropy_df.to_csv("entropy_df.csv", index=False)
    nmi_calc.nmi_df.to_csv("nmi_df.csv", index=False)
    graph_builder = GraphBuilder(
        topology, structure_calc.last_res_num, nmi_calc.nmi_df, graphdist
    )

    # Save the graph for future data science
    with open("graph.pkl", "wb") as pkl_file:
        pickle.dump(graph_builder.graph, pkl_file)

    # Per-edge confidence across replicas (multi-replica only, opt-in via -confidence)
    edge_conf_calc = None
    edge_conf_df = None
    if confidence and per_replica_dfs is not None:
        edge_conf_calc = EdgeConfidenceCalculator(per_replica_dfs)
        edge_conf_df = edge_conf_calc.build_edge_confidence_df()
        edge_conf_df_graph = EdgeConfidenceCalculator.filter_to_graph_edges(
            edge_conf_df, graph_builder.graph
        )
        edge_conf_df_graph.to_csv("edge_confidence.csv", index=False)
        print(
            "\033[1mPer-edge confidence across replicas saved to edge_confidence.csv\033[0m"
        )

    MDPathVisualize.visualise_graph(
        graph_builder.graph
    )  # Exports image of the Graph to PNG

    # Calculate paths
    path_total_weights = graph_builder.collect_path_total_weights(df_distant_residues)
    sorted_paths = sorted(path_total_weights, key=lambda x: x[1], reverse=True)
    with open("output.txt", "w") as file:
        for path, total_weight in sorted_paths[:numpath]:
            file.write(f"Path: {path}, Total Weight: {total_weight}\n")
    top_pathways = [path for path, _ in sorted_paths[:numpath]]

    # Calculate the paths including ligand interacting residues
    if lig_interaction:
        try:
            lig_interaction = [int(res) for res in lig_interaction]
        except ValueError:
            print("Error: All -lig inputs must be integers.")
        sorted_paths = [
            path
            for path in sorted_paths
            if any(residue in lig_interaction for residue in path[0])
        ]
        top_pathways = [path for path, _ in sorted_paths[:numpath]]
        print("\033[1mLigand pathways gathered..\033[0m")

    # Per-edge confidence for consecutive transitions along the top paths
    if confidence and edge_conf_df is not None:
        top_paths_conf_df = EdgeConfidenceCalculator.build_top_paths_edge_confidence_df(
            top_pathways, edge_conf_df
        )
        top_paths_conf_df.to_csv("top_paths_edge_confidence.csv", index=False)
        print(
            "\033[1mTop-path edge confidence saved to top_paths_edge_confidence.csv\033[0m"
        )

    # Bootstrap analysis
    if bootstrap:
        num_bootstrap_samples = int(bootstrap)
        bootstrap_analysis = BootstrapAnalysis(
            df_all_residues,
            df_distant_residues,
            sorted_paths,
            num_bootstrap_samples,
            numpath,
            topology,
            structure_calc.last_res_num,
            graphdist,
        )
        file_name = "path_confidence_intervals.txt"
        bootstrap_analysis.bootstrap_write(file_name)
        print(f"Path confidence intervals have been saved to {file_name}")

    # Cluster pathways to get signaltransduction paths
    clustering = PatwayClustering(df_close_res, top_pathways, num_parallel_processes)
    clusters = clustering.pathways_cluster()
    cluster_pathways_dict = clustering.pathway_clusters_dictionary(
        clusters, sorted_paths
    )
    residue_coordinates_dict = MDPathVisualize.residue_CA_coordinates(
        "first_frame.pdb", structure_calc.last_res_num
    )

    # Export residue coordinates and pathways dict for comparisson functionality
    with open("residue_coordinates.pkl", "wb") as pkl_file:
        pickle.dump(residue_coordinates_dict, pkl_file)

    with open("cluster_pathways_dict.pkl", "wb") as pkl_file:
        pickle.dump(cluster_pathways_dict, pkl_file)

    with open("top_pathways.pkl", "wb") as pkl_file:
        pickle.dump(top_pathways, pkl_file)

    # Export the cluster pathways for visualization
    updated_dict = MDPathVisualize.apply_backtracking(
        cluster_pathways_dict, residue_coordinates_dict
    )
    formated_dict = MDPathVisualize.format_dict(updated_dict)
    with open("clusters_paths.json", "w") as json_file:
        json.dump(formated_dict, json_file)
    path_properties = MDPathVisualize.precompute_path_properties(formated_dict)
    with open("precomputed_clusters_paths.json", "w") as out_file:
        json.dump(path_properties, out_file, indent=4)
    quick_path_properties = MDPathVisualize.precompute_cluster_properties_quick(
        formated_dict
    )
    with open("quick_precomputed_clusters_paths.json", "w") as out_file2:
        json.dump(quick_path_properties, out_file2, indent=4)

    if spline:
        MDPathVisualize.create_splines("quick_precomputed_clusters_paths.json")


# if water:
#     from mdpath.src.water_tracing import WaterTracer
#     water_tracer = WaterTracer(
#         topology=topology,
#         trajectory=trajectory,
#         path_total_weights=path_total_weights,
#         occurrence_threshold=float(water) / 100.0,
#     )
#     water_tracer.pathway_water_data.save("pathway_water_data.pkl")
#     water_tracer.pathway_water_data.to_dataframe().to_csv(
#         "pathway_water_bridges.csv", index=False
#     )


if __name__ == "__main__":
    main()
