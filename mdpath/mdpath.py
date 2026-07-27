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
import MDAnalysis as mda
import json
import pickle

from mdpath.src.structure import StructureCalculations, DihedralAngles
from mdpath.src.mutual_information import NMICalculator, binning_diagnostic
from mdpath.src.graph import GraphBuilder
from mdpath.src.cluster import PatwayClustering
from mdpath.src.visualization import MDPathVisualize
from mdpath.src.bootstrap import BootstrapAnalysis
from mdpath.src.confidence import EdgeConfidenceCalculator
from mdpath.src.null_model import BlockSplitNull
from mdpath.src.path_confidence_viz import ConfidencePathVisualizer


def _select_subsystem(topology, trajectories, selection, prefix, label):
    """Restrict topology and every replica trajectory to an atom selection.

    Writes ``selected_<prefix>.pdb`` and ``selected_<prefix>_<i>.dcd`` per replica.

    Returns:
        tuple: (new_topology, new_trajectories, universe) for the selected subsystem.
    """
    universe = mda.Universe(topology, trajectories[0])
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"No atoms found for {selection}")
    merged = mda.Merge(atoms)
    out_topology = f"selected_{prefix}.pdb"
    merged.atoms.write(out_topology)

    new_trajectories = []
    for i, traj_file in enumerate(trajectories):
        u = mda.Universe(topology, traj_file)
        sel = u.select_atoms(selection)
        out_name = f"selected_{prefix}_{i}.dcd"
        with mda.Writer(out_name, sel.n_atoms) as W:
            for ts in u.trajectory:
                merged.atoms.positions = sel.positions
                W.write(merged.atoms)
        new_trajectories.append(out_name)

    print(
        f"{label} selected for {len(new_trajectories)} trajectory file(s) and will now be analyzed."
    )
    return (
        out_topology,
        new_trajectories,
        mda.Universe(out_topology, new_trajectories[0]),
    )


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
        "Produces edge_confidence.csv (all graph edges), top_paths_edge_confidence.csv "
        "(consecutive transitions along the top pathways), and confidence_paths_chimerax.py "
        "(a self-contained ChimeraX viewer of the top paths colored by confidence).",
        action="store_true",
    )
    parser.add_argument(
        "-null",
        dest="null_model",
        choices=("none", "block"),
        default="none",
        help="Contiguous block-split reference for replica confidence.",
    )
    parser.add_argument("-null-block-ns", type=float, default=100.0)
    parser.add_argument("-null-block-sweep", default="25,50,100,250")
    parser.add_argument(
        "-null-sources",
        default="all",
        help="all or comma-separated 1-based replica IDs",
    )
    parser.add_argument("-null-draws", type=int, default=100)
    parser.add_argument("-null-seed", type=int, default=0)
    parser.add_argument("-ddof", type=int, choices=(0, 1), default=0)
    parser.add_argument("-null-outdir", default="null_analysis")
    parser.add_argument(
        "-hist-range",
        dest="hist_range",
        nargs="+",
        default=["-180", "180"],
        help="Fixed histogram support for the dihedral-movement histograms, in "
        "degrees, as two floats (default: -180 180). Fixing the bin edges is what "
        "makes NMI comparable across replicas, blocks and bootstrap samples. Pass "
        '"none" to restore the pre-fix behaviour, where each input derived its own '
        "support from its own extrema.",
    )
    parser.add_argument(
        "-diagnose-bins",
        dest="diagnose_bins",
        action="store_true",
        help="Write binning_diagnostic.csv: per-residue histogram occupancy under "
        "the chosen support and branch-cut exposure (frames with |dphi| > 180 "
        "before wrapping). Use it to check whether the fixed support is too wide "
        "for the observed dihedral movements.",
    )
    parser.add_argument(
        "-experimental-c-corr",
        dest="experimental_c_corr",
        action="store_true",
        help="Also write an EXPERIMENTAL c_corr-colored ChimeraX viewer (opt-in; "
        "c_corr is an unvalidated heuristic, rho is the primary comparison).",
    )

    args = parser.parse_args()
    # -hist-range: two floats, or the literal "none" to restore data-derived binning.
    if len(args.hist_range) == 1 and str(args.hist_range[0]).lower() == "none":
        hist_range = None
    elif len(args.hist_range) == 2:
        hist_range = (float(args.hist_range[0]), float(args.hist_range[1]))
        if hist_range[0] >= hist_range[1]:
            raise ValueError("-hist-range needs an increasing (low, high) pair")
    else:
        raise ValueError('-hist-range takes two floats or the literal "none"')
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
    trajectory_dt_ps = float(traj.trajectory.dt)
    # Capture every replica's timestep from the ORIGINAL inputs: a later -chain or
    # -segid selection rewrites the trajectories, and those derived files carry a
    # default timestep, which would make the block-reference check vacuous.
    replica_dts = [
        float(mda.Universe(topology, traj_file).trajectory.dt)
        for traj_file in trajectories
    ]
    lig_interaction = args.lig_interaction
    bootstrap = args.bootstrap
    fardist = float(args.fardist)
    closedist = float(args.closedist)
    graphdist = float(args.graphdist)
    numpath = int(args.numpath)
    invert = bool(args.invert)
    spline = bool(args.spline)
    confidence = bool(args.confidence) or args.null_model != "none"
    if confidence and len(trajectories) < 2:
        print(
            "Error: -confidence requires multiple replica trajectories "
            "(-traj traj1 traj2 ...)."
        )
        exit()

    # Chain selection
    if args.chain:
        chain = str(args.chain)
        topology, selected, traj = _select_subsystem(
            topology, trajectories, f"chainID {chain}", "chain", f"Chain {chain}"
        )
        trajectories = selected

    # Segment ID selection (CHARMM compatibility)
    if args.segid:
        segid = str(args.segid)
        topology, selected, traj = _select_subsystem(
            topology, trajectories, f"segid {segid}", "segid", f"Segment {segid}"
        )
        trajectories = selected

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
    if graphdist == closedist:
        graph_pairs = list(df_close_res.itertuples(index=False, name=None))
    else:
        graph_pairs = list(
            structure_calc.calculate_residue_suroundings(graphdist, "close").itertuples(
                index=False, name=None
            )
        )

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

    if args.diagnose_bins:
        binning_diagnostic(
            df_all_residues,
            num_bins=35,
            hist_range=hist_range,
            csv_path="binning_diagnostic.csv",
        )

    # Calculate the mutual information and build the graph
    nmi_calc = NMICalculator(df_all_residues, invert=invert, hist_range=hist_range)
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
    null_outputs = None
    if confidence and per_replica_dfs is not None:
        edge_conf_calc = EdgeConfidenceCalculator(
            per_replica_dfs,
            pairs=graph_pairs,
            ddof=args.ddof,
            hist_range=hist_range,
        )
        edge_conf_df = edge_conf_calc.build_edge_confidence_df()
        edge_conf_df_graph = EdgeConfidenceCalculator.filter_to_graph_edges(
            edge_conf_df, graph_builder.graph
        )
        edge_conf_df_graph.to_csv("edge_confidence.csv", index=False)
        print(
            "\033[1mPer-edge confidence across replicas saved to edge_confidence.csv\033[0m"
        )
        if args.null_model == "block":
            if args.null_sources == "all":
                source_ids = list(range(len(per_replica_dfs)))
            else:
                source_ids = [int(value) - 1 for value in args.null_sources.split(",")]
                if (
                    not source_ids
                    or min(source_ids) < 0
                    or max(source_ids) >= len(per_replica_dfs)
                ):
                    raise ValueError(
                        "-null-sources contains an invalid 1-based replica ID"
                    )
            # The ns->frame conversion assumes a shared timestep; validate the
            # timesteps captured from the original inputs (see above).
            if max(replica_dts) - min(replica_dts) > 1e-6 * max(replica_dts):
                raise ValueError(
                    "block null requires a shared timestep across replicas; got dt(ps) "
                    f"in [{min(replica_dts):g}, {max(replica_dts):g}]. Re-sample the "
                    "trajectories to a common dt or run the sources separately."
                )
            sweep = [
                float(value)
                for value in args.null_block_sweep.split(",")
                if value.strip()
            ]
            if args.null_block_ns not in sweep:
                sweep.append(args.null_block_ns)
            null_calc = BlockSplitNull(
                [per_replica_dfs[index] for index in source_ids],
                per_replica_dfs,
                block_lengths_ns=sweep,
                frames_per_ns=1000.0 / trajectory_dt_ps,
                pairs=graph_pairs,
                draws=args.null_draws,
                seed=args.null_seed,
                ddof=args.ddof,
                n_jobs=num_parallel_processes,
                hist_range=hist_range,
            )
            null_outputs = null_calc.run(args.null_outdir, args.null_block_ns)
            status = null_outputs["edge_null_confidence"].corr_status.value_counts(
                normalize=True
            )
            print(
                "Block reference (rho is the primary comparison): "
                f"{status.get('above_reference', 0.0):.1%} above-reference (resolved), "
                f"{status.get('below_or_equal_reference', 0.0):.1%} below/equal, "
                f"{status.get('undefined', 0.0):.1%} undefined "
                "-> experimental c_corr is NaN for all but the resolved edges."
            )
            # Anchor the fraction: if the observed windows and the reference blocks
            # were exchangeable, ~50% of edges would land above the reference. This
            # is a descriptive comparison, not a controlled false-positive rate.
            print(
                "  (chance expectation under exchangeability is ~50%; this is a "
                "descriptive fraction, not a multiplicity-controlled error rate)"
            )

    MDPathVisualize.visualise_graph(
        graph_builder.graph
    )  # Exports image of the Graph to PNG

    # Calculate paths
    path_total_weights = graph_builder.collect_path_total_weights_parallel(
        df_distant_residues, num_parallel_processes
    )
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
            hist_range=hist_range,
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

    # Self-contained ChimeraX viewer of the top paths colored by replica
    # confidence (auto-scaled tube thickness, hover shows the value).
    if confidence and edge_conf_df is not None:
        edge_null_df = (
            null_outputs["edge_null_confidence"] if null_outputs is not None else None
        )
        try:
            conf_vis = ConfidencePathVisualizer.write_chimerax_script(
                top_pathways,
                residue_coordinates_dict,
                edge_conf_df,
                pdb_file="first_frame.pdb",
                out_path="confidence_paths_chimerax.py",
                edge_null_df=edge_null_df,
            )
            print(
                "\033[1mChimeraX confidence viewer saved to confidence_paths_chimerax.py "
                "({0} paths, tube radius {1:.2f}-{2:.2f} A). Drag it onto a ChimeraX window.\033[0m".format(
                    conf_vis["paths"], conf_vis["radius_min"], conf_vis["radius_max"]
                )
            )
        except Exception as exc:
            print(f"Could not generate ChimeraX confidence viewer: {exc}")

        # Optional second viewer, identical hover but colored by the EXPERIMENTAL
        # matched-window c_corr heuristic. This is opt-in (-experimental-c-corr)
        # because c_corr is an unvalidated exploratory score; rho and the
        # production-colored viewer remain the primary outputs.
        if edge_null_df is not None and args.experimental_c_corr:
            try:
                corr_vis = ConfidencePathVisualizer.write_chimerax_script(
                    top_pathways,
                    residue_coordinates_dict,
                    edge_conf_df,
                    pdb_file="first_frame.pdb",
                    out_path="confidence_paths_chimerax_experimental_c_corr.py",
                    edge_null_df=edge_null_df,
                    color_by="c_corr",
                )
                print(
                    "\033[1mExperimental c_corr ChimeraX viewer saved to "
                    "confidence_paths_chimerax_experimental_c_corr.py "
                    "({0} paths; EXPERIMENTAL matched-window heuristic, "
                    "undefined/below-reference edges gray).\033[0m".format(
                        corr_vis["paths"]
                    )
                )
            except Exception as exc:
                print(f"Could not generate experimental c_corr ChimeraX viewer: {exc}")

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


if __name__ == "__main__":
    main()
