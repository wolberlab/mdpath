import os
import sys
import glob
import pytest
from io import StringIO
from mdpath.mdpath_tools import edit_3D_visualization_json
from mdpath.mdpath_tools import path_comparison
from mdpath.mdpath_tools import multitraj_analysis


def test_gpcr_2D_vis(tmp_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_directory = os.getcwd()
    topology = os.path.join(script_dir, "test_topology.pdb")
    cluster_file = os.path.join(script_dir, "cluster_pathways_dict_tools.pkl")
    cutoff_percentage = "1"

    sys.argv = [
        "mdpath_gpcr",
        "-top",
        topology,
        "-clust",
        cluster_file,
        "-cut",
        cutoff_percentage,
    ]

    original_stdout = sys.stdout
    sys.stdout = StringIO()

    try:

        from mdpath.mdpath_tools import gpcr_2D_vis

        with pytest.raises(SystemExit) as exc_info:
            gpcr_2D_vis()

        assert exc_info.value.code == 0

        generated_files = glob.glob(
            os.path.join(current_directory, "GPCR_2D_pathways_cluster*")
        )
        assert (
            len(generated_files) > 0
        ), "No GPCR_2D_pathways_cluster files were generated."

    finally:
        sys.stdout = original_stdout


def test_edit_3D_visualization_json(tmp_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    recolor = os.path.join(script_dir, "easy_read_colors.json")
    json_file = os.path.join(script_dir, "quick_precomputed_clusters_paths_tools.json")

    expected_message = "Recoloring requires a valid -json to recolor."

    original_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        sys.argv = [
            "mdpath_json_editor",
            "-recolor",
            recolor,
        ]
        with pytest.raises(SystemExit) as exc_info:
            edit_3D_visualization_json()
        output = sys.stdout.getvalue()
        assert expected_message in output

        sys.argv = ["mdpath_json_editor", "-recolor", recolor, "-json", json_file]
        with pytest.raises(SystemExit) as exc_info:
            edit_3D_visualization_json()
        assert exc_info.value.code == 0

        generated_files = glob.glob(os.path.join(script_dir, "*_recolored*"))
        assert len(generated_files) > 0, "No recolored json file was generated."

        os.remove(
            os.path.join(
                script_dir, "quick_precomputed_clusters_paths_tools_recolored.json"
            )
        )

        sys.argv = [
            "mdpath_json_editor",
            "-scale",
            "3",
            "-json",
            json_file,
            "-flat",
            "3",
        ]
        with pytest.raises(SystemExit) as exc_info:
            edit_3D_visualization_json()
        generated_files = glob.glob(os.path.join(script_dir, "*_scaled_*"))
        assert len(generated_files) == 0, "Files should not be created!"

        sys.argv = [
            "mdpath_json_editor",
            "-scale",
            "3",
            "-json",
            json_file,
            "-clusterscale",
            "3",
        ]
        with pytest.raises(SystemExit) as exc_info:
            edit_3D_visualization_json()
        generated_files = glob.glob(os.path.join(script_dir, "*_scaled_*"))
        assert len(generated_files) == 0, "Files should not be created!"

        sys.argv = [
            "mdpath_json_editor",
            "-scale",
            "3",
            "-json",
            json_file,
        ]
        with pytest.raises(SystemExit) as exc_info:
            edit_3D_visualization_json()
        assert exc_info.value.code == 0
        generated_files = glob.glob(os.path.join(script_dir, "*_scaled_*"))
        assert len(generated_files) > 0, "No rescaled json file was generated."

        os.remove(
            os.path.join(
                script_dir, "quick_precomputed_clusters_paths_tools_scaled_3.0.json"
            )
        )

        sys.argv = [
            "mdpath_json_editor",
            "-flat",
            "3",
            "-json",
            json_file,
            "-scale",
            "3",
        ]
        with pytest.raises(SystemExit) as exc_info:
            edit_3D_visualization_json()
        generated_files = glob.glob(os.path.join(script_dir, "*_flat_*"))
        assert len(generated_files) == 0, "Files should not be created!"

        sys.argv = [
            "mdpath_json_editor",
            "-flat",
            "3",
            "-json",
            json_file,
            "-clusterscale",
            "3",
        ]
        with pytest.raises(SystemExit) as exc_info:
            edit_3D_visualization_json()
        generated_files = glob.glob(os.path.join(script_dir, "*_flat_*"))
        assert len(generated_files) == 0, "Files should not be created!"

        sys.argv = [
            "mdpath_json_editor",
            "-flat",
            "3",
            "-json",
            json_file,
        ]
        with pytest.raises(SystemExit) as exc_info:
            edit_3D_visualization_json()
        assert exc_info.value.code == 0
        generated_files = glob.glob(os.path.join(script_dir, "*_flat_*"))
        assert len(generated_files) > 0, "No flattened json file was generated."

        os.remove(
            os.path.join(
                script_dir, "quick_precomputed_clusters_paths_tools_flat_3.0.json"
            )
        )

        sys.argv = [
            "mdpath_json_editor",
            "-clusterscale",
            "3",
            "-json",
            json_file,
            "-flat",
            "3",
        ]
        with pytest.raises(SystemExit) as exc_info:
            edit_3D_visualization_json()
        generated_files = glob.glob(os.path.join(script_dir, "*_cluster_scaled_*"))
        assert len(generated_files) == 0, "Files should not be created!"

        sys.argv = [
            "mdpath_json_editor",
            "-clusterscale",
            "3",
            "-json",
            json_file,
            "-scale",
            "3",
        ]
        with pytest.raises(SystemExit) as exc_info:
            edit_3D_visualization_json()
        generated_files = glob.glob(os.path.join(script_dir, "*_cluster_scaled_*"))
        assert len(generated_files) == 0, "Files should not be created!"

        sys.argv = ["mdpath_json_editor", "-clusterscale", "3", "-json", json_file]
        with pytest.raises(SystemExit) as exc_info:
            edit_3D_visualization_json()
        assert exc_info.value.code == 0
        generated_files = glob.glob(os.path.join(script_dir, "*_cluster_scaled_*"))
        assert len(generated_files) > 0, "No cluster-scaled json file was generated."

        os.remove(
            os.path.join(
                script_dir,
                "quick_precomputed_clusters_paths_tools_cluster_scaled_3.0.json",
            )
        )

    finally:
        sys.stdout = original_stdout


def test_path_comparison():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_directory = os.getcwd()
    atop = os.path.join(script_dir, "residue_coordinates_tools.pkl")
    bcluster = os.path.join(script_dir, "cluster_pathways_dict_testpc.pkl")
    expected_message = "Topology (residue_coordinates) and bcluster (cluster) are required and a json needed for comparing two simulations."

    original_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        sys.argv = ["mdpath_compare", "-atop", atop]
        with pytest.raises(SystemExit) as exc_info:
            path_comparison()
        output = sys.stdout.getvalue()
        assert expected_message in output

        sys.argv = ["mdpath_compare", "-bcluster", bcluster]
        with pytest.raises(SystemExit) as exc_info:
            path_comparison()
        output = sys.stdout.getvalue()
        assert expected_message in output

        sys.argv = ["mdpath_compare", "-atop", atop, "-bcluster", bcluster]
        with pytest.raises(SystemExit) as exc_info:
            path_comparison()
        assert exc_info.value.code == 0

        generated_files = glob.glob(
            os.path.join(current_directory, "morphed_clusters_paths.json")
        )
        assert (
            len(generated_files) > 0
        ), "No morphed_clusters_paths.json file was generated."

    finally:
        sys.stdout = original_stdout
        for file_path in generated_files:
            os.remove(file_path)


def test_multitraj_analysis():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_directory = os.getcwd()
    topology = os.path.join(script_dir, "multitraj.pdb")
    multitraj_1 = os.path.join(script_dir, "top_pathways.pkl")

    original_stdout = sys.stdout
    sys.stdout = StringIO()

    generated_files = []  # Define an empty list for generated_files

    try:
        sys.argv = [
            "mdpath_multitraj",
            "-top",
            topology,
            "-multitraj",
            multitraj_1,
            multitraj_1,
        ]
        with pytest.raises(SystemExit) as exc_info:
            multitraj_analysis()

        assert exc_info.value.code == 0, "The command failed with non-zero exit code"

        generated_files = glob.glob(
            os.path.join(current_directory, "multitraj_clusters_paths.json")
        )
        assert (
            len(generated_files) > 0
        ), "No multitraj_clusters_paths.json file was generated."

    finally:
        sys.stdout = original_stdout
        for file_path in generated_files:
            os.remove(file_path)
