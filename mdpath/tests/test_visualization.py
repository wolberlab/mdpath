import pytest
import tempfile
import os
import numpy as np
import networkx as nx
from unittest.mock import MagicMock, patch, mock_open
from mdpath.src.visualization import MDPathVisualize
import MDAnalysis as mda
from io import StringIO
import sys
import glob


def create_mock_pdb(content: str) -> str:
    """Helper function to create a temporary PDB file."""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    with open(tmp_file.name, "w") as f:
        f.write(content)
    return tmp_file.name


def are_vectors_close(vec1, vec2, tol=1e-6):
    return np.allclose(vec1, vec2, atol=tol)


def test_residue_CA_coordinates():
    pdb_content = """
ATOM      1  N   SER R  66     163.079 132.512 139.525  1.00 67.17           N
ATOM      2  CA  SER R  66     162.030 133.517 139.402  1.00 67.17           C
ATOM      3  C   SER R  66     161.837 133.927 137.947  1.00 67.17           C
ATOM      4  O   SER R  66     160.746 133.785 137.394  1.00 67.17           O
ATOM      5  CB  SER R  66     162.361 134.744 140.256  1.00 67.17           C
ATOM      6  OG  SER R  66     163.513 135.410 139.767  1.00 67.17           O
ATOM      7  N   MET R  67     162.906 134.456 137.345  1.00 67.62           N
ATOM      8  CA  MET R  67     162.863 134.856 135.941  1.00 67.62           C
ATOM      9  C   MET R  67     162.712 133.650 135.020  1.00 67.62           C
ATOM     10  O   MET R  67     161.925 133.688 134.064  1.00 67.62           O
ATOM     11  CB  MET R  67     164.124 135.646 135.590  1.00 67.62           C
ATOM     12  CG  MET R  67     164.130 136.230 134.189  1.00 67.62           C
ATOM     13  SD  MET R  67     162.847 137.472 133.944  1.00 67.62           S
ATOM     14  CE  MET R  67     163.512 138.833 134.901  1.00 67.62           C
ATOM     15  N   ILE R  68     163.445 132.569 135.314  1.00 62.30           N
ATOM     16  CA  ILE R  68     163.415 131.357 134.496  1.00 62.30           C
ATOM     17  C   ILE R  68     162.033 130.714 134.535  1.00 62.30           C
ATOM     18  O   ILE R  68     161.528 130.240 133.509  1.00 62.30           O
ATOM     19  CB  ILE R  68     164.513 130.386 134.971  1.00 62.30           C
ATOM     20  CG1 ILE R  68     165.891 131.038 134.844  1.00 62.30           C
ATOM     21  CG2 ILE R  68     164.493 129.087 134.176  1.00 62.30           C
ATOM     22  CD1 ILE R  68     166.997 130.265 135.529  1.00 62.30           C
ATOM     23  N   THR R  69     161.394 130.716 135.710  1.00 57.94           N
ATOM     24  CA  THR R  69     160.061 130.137 135.858  1.00 57.94           C
ATOM     25  C   THR R  69     159.027 130.903 135.036  1.00 57.94           C
ATOM     26  O   THR R  69     158.209 130.295 134.331  1.00 57.94           O
ATOM     27  CB  THR R  69     159.673 130.122 137.339  1.00 57.94           C
TER
"""
    pdb_file = create_mock_pdb(pdb_content)
    end_residue = 69

    expected_result = {
        66: [[162.030, 133.517, 139.402]],
        67: [[162.863, 134.856, 135.941]],
        68: [[163.415, 131.357, 134.496]],
        69: [[160.061, 130.137, 135.858]],
    }

    result = MDPathVisualize.residue_CA_coordinates(pdb_file, end_residue)
    result = {k: [list(v) for v in vs] for k, vs in result.items()}

    for res_id, coords_list in expected_result.items():
        assert res_id in result, f"Residue {res_id} not found in result"
        for expected_coords, actual_coords in zip(coords_list, result[res_id]):
            assert are_vectors_close(
                expected_coords, actual_coords
            ), f"Expected {expected_coords} but got {actual_coords} for residue {res_id}"


def test_apply_backtracking():
    original_dict = {
        "cluster1": [[1, 2], [3, 4]],
        "cluster2": [[5, 6], [7, 8]],
        "cluster3": [[9, 10]],
    }
    translation_dict = {1: "A", 2: "B", 5: "E", 6: "F", 10: "J"}
    expected = {
        "cluster1": [["A", "B"], [3, 4]],
        "cluster2": [["E", "F"], [7, 8]],
        "cluster3": [[9, "J"]],
    }

    result = MDPathVisualize.apply_backtracking(original_dict, translation_dict)
    assert result == expected


def test_cluster_prep_for_visualisation():
    pdb_file = "mock_pdb_file.pdb"
    input_cluster = [[1, 2], [3]]
    mock_coordinates = {1: (1.0, 1.0, 1.0), 2: (2.0, 2.0, 2.0), 3: (3.0, 3.0, 3.0)}

    with patch("Bio.PDB.PDBParser") as mock_parser:
        mock_structure = MagicMock()

        def get_structure(name, file):
            return mock_structure

        mock_parser.return_value.get_structure.side_effect = get_structure

        mock_residues = {}
        for residue_id, coord in mock_coordinates.items():
            mock_residue = MagicMock()
            mock_atom = MagicMock()
            mock_atom.get_coord.return_value = coord
            mock_residue.__getitem__.return_value = mock_atom
            mock_residues[("", residue_id, "")] = mock_residue

        def getitem(res_id):
            if res_id in mock_residues:
                return mock_residues[res_id]
            else:
                raise KeyError

        mock_structure[0].__getitem__.side_effect = getitem

        result = MDPathVisualize.cluster_prep_for_visualisation(input_cluster, pdb_file)

        expected_result = [[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)], [(3.0, 3.0, 3.0)]]

        assert result == expected_result


def test_format_dict():
    input_dict = {"array": np.array([1, 2, 3]), "nested_list": [1, 2, np.array([3, 4])]}
    expected_output = {"array": [1, 2, 3], "nested_list": [1, 2, [3, 4]]}
    assert MDPathVisualize.format_dict(input_dict) == expected_output
    assert MDPathVisualize.format_dict({}) == {}
    input_dict = {"nested": [1, [2, 3], np.array([4, 5])]}
    expected_output = {"nested": [1, [2, 3], [4, 5]]}
    assert MDPathVisualize.format_dict(input_dict) == expected_output
    input_dict = {"mixed": [1, "string", np.array([6, 7])]}
    expected_output = {"mixed": [1, "string", [6, 7]]}
    assert MDPathVisualize.format_dict(input_dict) == expected_output


def test_visualise_graph():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])

    MDPathVisualize.visualise_graph(G)

    assert os.path.exists("graph.png"), "graph.png file was not created."

    if os.path.exists("graph.png"):
        os.remove("graph.png")


def test_precompute_path_properties():
    json_data = {
        "cluster1": [
            [[[1, 2, 3]], [[4, 5, 6]]],
            [[[7, 8, 9]], [[10, 11, 12]]],
        ],
        "cluster2": [
            [[[13, 14, 15]], [[16, 17, 18]]],
        ],
    }

    expected_output = [
        {
            "clusterid": "cluster1",
            "pathway_index": 0,
            "path_segment_index": 0,
            "coord1": [1, 2, 3],
            "coord2": [4, 5, 6],
            "color": [0.1216, 0.4667, 0.7059],
            "radius": 0.015,
            "path_number": 1,
        },
        {
            "clusterid": "cluster1",
            "pathway_index": 1,
            "path_segment_index": 0,
            "coord1": [7, 8, 9],
            "coord2": [10, 11, 12],
            "color": [0.1216, 0.4667, 0.7059],
            "radius": 0.015,
            "path_number": 2,
        },
        {
            "clusterid": "cluster2",
            "pathway_index": 0,
            "path_segment_index": 0,
            "coord1": [13, 14, 15],
            "coord2": [16, 17, 18],
            "color": [0.1725, 0.6647, 0.1725],
            "radius": 0.015,
            "path_number": 1,
        },
    ]

    result = MDPathVisualize.precompute_path_properties(json_data)
    assert result == expected_output


def test_precompute_cluster_properties_quick():
    json_data = {
        "cluster1": [[[[1, 2, 3]], [[4, 5, 6]]], [[[1, 2, 3]], [[4, 5, 6]]]],
        "cluster2": [[[[7, 8, 9]], [[10, 11, 12]]]],
    }

    expected_output = [
        {
            "clusterid": "cluster1",
            "coord1": [1, 2, 3],
            "coord2": [4, 5, 6],
            "color": [0.1216, 0.4667, 0.7059],
            "radius": 0.015,
        },
        {
            "clusterid": "cluster1",
            "coord1": [1, 2, 3],
            "coord2": [4, 5, 6],
            "color": [0.1216, 0.4667, 0.7059],
            "radius": 0.03,
        },
        {
            "clusterid": "cluster2",
            "coord1": [7, 8, 9],
            "coord2": [10, 11, 12],
            "color": [0.1725, 0.6647, 0.1725],
            "radius": 0.015,
        },
    ]
    actual_output = MDPathVisualize.precompute_cluster_properties_quick(json_data)
    assert (
        actual_output == expected_output
    ), f"Expected {expected_output}, but got {actual_output}"


@patch("mdpath.src.visualization.mda.Universe")
def test_remove_non_protein(mock_universe):
    mock_sys = MagicMock()
    mock_universe.return_value = mock_sys
    mock_protein_atoms = MagicMock()
    mock_sys.select_atoms.return_value = mock_protein_atoms
    input_pdb = "mock_input.pdb"
    output_pdb = "mock_output.pdb"
    MDPathVisualize.remove_non_protein(input_pdb, output_pdb)
    mock_universe.assert_called_once_with(input_pdb)
    mock_sys.select_atoms.assert_called_once_with("protein")
    mock_protein_atoms.write.assert_called_once_with(output_pdb)


@patch("mdpath.src.visualization.requests.post")
@patch("builtins.open", new_callable=mock_open)
def test_assign_generic_numbers(mock_open_file, mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "mock_pdb_content"
    mock_post.return_value = mock_response

    pdb_file_path = "mock_input.pdb"
    output_file_path = "mock_output.pdb"
    MDPathVisualize.assign_generic_numbers(pdb_file_path, output_file_path)

    mock_post.assert_called_once_with(
        "https://gpcrdb.org/services/structure/assign_generic_numbers",
        files={"pdb_file": mock_open_file.return_value},
    )
    mock_open_file.assert_any_call(pdb_file_path, "rb")
    mock_open_file.assert_any_call(output_file_path, "w")
    mock_open_file().write.assert_called_once_with("mock_pdb_content")

def test_mdpath_spline(tmp_path):
    current_directory = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, "quick_precomputed_clusters_paths_tools.json")

    sys.argv = [
        "mdpath_spline",
        "-json",
        json_file,
    
    ]
    original_stdout = sys.stdout
    sys.stdout = StringIO()

    try:

        from mdpath.mdpath_tools import spline
    
        spline()
   
        cluster_meshes_directory = os.path.join(current_directory, "cluster_meshes")
        if os.path.isdir(cluster_meshes_directory):
            generated_files = glob.glob(os.path.join(cluster_meshes_directory, "*"))
        else:
            generated_files = []
            assert len(generated_files) > 0, "No files were generated in the cluster_meshes directory."

    finally:
        sys.stdout = original_stdout

