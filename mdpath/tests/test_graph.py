import pytest
import networkx as nx
import pandas as pd
from unittest.mock import patch, MagicMock
from mdpath.src.graph import GraphBuilder
import os


def test_graph_builder_initialization():
    pdb = "test_topology.pdb"
    last_residue = 100
    mi_diff_df = pd.DataFrame(
        {"residue1": [1, 2, 3], "residue2": [4, 5, 6], "distance": [1.2, 2.3, 3.4]}
    )
    graphdist = 10

    with patch.object(
        GraphBuilder, "graph_builder", return_value="mocked_graph"
    ) as mock_graph_builder:
        gb = GraphBuilder(pdb, last_residue, mi_diff_df, graphdist)
        assert gb.pdb == pdb
        assert gb.end == last_residue
        assert gb.mi_diff_df.equals(mi_diff_df)
        assert gb.dist == graphdist

        mock_graph_builder.assert_called_once()
        assert gb.graph == "mocked_graph"


def test_max_weight_shortest_path():
    with (
        patch("mdpath.src.graph.StructureCalculations"),
        patch("mdpath.src.graph.PDB.PDBParser"),
    ):
        G = nx.Graph()
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=2.0)
        G.add_edge(1, 3, weight=4.0)
        G.add_edge(3, 4, weight=1.0)
        G.add_edge(2, 4, weight=3.0)

        graph_builder = GraphBuilder(
            pdb="", last_residue=0, mi_diff_df=pd.DataFrame(), graphdist=5
        )
        graph_builder.graph = G

        source = 1
        target = 4
        expected_path = [1, 3, 4]
        expected_weight = 5.0
        path, weight = graph_builder.max_weight_shortest_path(source, target)
        assert path == expected_path, f"Expected path: {expected_path}, but got: {path}"
        assert weight == pytest.approx(
            expected_weight
        ), f"Expected weight: {expected_weight}, but got: {weight}"

        source = 1
        target = 3
        expected_path = [1, 3]
        expected_weight = 4.0
        path, weight = graph_builder.max_weight_shortest_path(source, target)
        assert path == expected_path, f"Expected path: {expected_path}, but got: {path}"
        assert weight == pytest.approx(
            expected_weight
        ), f"Expected weight: {expected_weight}, but got: {weight}"

        source = 2
        target = 4
        expected_path = [2, 4]
        expected_weight = 3.0
        path, weight = graph_builder.max_weight_shortest_path(source, target)
        assert path == expected_path, f"Expected path: {expected_path}, but got: {path}"
        assert weight == pytest.approx(
            expected_weight
        ), f"Expected weight: {expected_weight}, but got: {weight}"

        source = 1
        target = 2
        expected_path = [1, 2]
        expected_weight = 1.0
        path, weight = graph_builder.max_weight_shortest_path(source, target)
        assert path == expected_path, f"Expected path: {expected_path}, but got: {path}"
        assert weight == pytest.approx(
            expected_weight
        ), f"Expected weight: {expected_weight}, but got: {weight}"


def test_graph_assign_weights():
    with (
        patch("mdpath.src.graph.StructureCalculations"),
        patch("mdpath.src.graph.PDB.PDBParser"),
    ):
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])

        data = {
            "Residue Pair": [("Res 1", "Res 2"), ("Res 2", "Res 3")],
            "MI Difference": [0.5, 0.8],
        }
        mi_diff_df = pd.DataFrame(data)

        graph_builder = GraphBuilder(
            pdb="", last_residue=0, mi_diff_df=mi_diff_df, graphdist=5
        )
        G_weighted = graph_builder.graph_assign_weights(G)

        expected_weights = {(1, 2): 0.5, (2, 3): 0.8}
        for edge, weight in expected_weights.items():
            assert (
                "weight" in G_weighted.edges[edge]
            ), f"Weight for edge {edge} not found"
            assert (
                G_weighted.edges[edge]["weight"] == weight
            ), f"Weight for edge {edge} is not {weight}"

        for edge in G_weighted.edges:
            if edge not in expected_weights:
                assert (
                    "weight" not in G_weighted.edges[edge]
                ), f"Unexpected weight for edge {edge}"


def test_collect_path_total_weights():
    with (
        patch("mdpath.src.graph.StructureCalculations"),
        patch("mdpath.src.graph.PDB.PDBParser"),
    ):
        G = nx.Graph()
        G.add_edge(1, 2, weight=10)
        G.add_edge(2, 3, weight=20)
        G.add_edge(1, 3, weight=15)

        df = pd.DataFrame({"Residue1": [1, 2], "Residue2": [3, 4]})

        graph_builder = GraphBuilder(
            pdb="", last_residue=0, mi_diff_df=pd.DataFrame(), graphdist=5
        )
        graph_builder.graph = G

        test_cases = [
            {
                "mock_side_effect": [([1, 2, 3], 30), nx.NetworkXNoPath],
                "expected_result": [([1, 2, 3], 30)],
            },
            {
                "mock_side_effect": [nx.NetworkXNoPath, nx.NetworkXNoPath],
                "expected_result": [],
            },
            {
                "mock_side_effect": [([1, 2, 3], 30), nx.NetworkXNoPath],
                "df": pd.DataFrame(columns=["Residue1", "Residue2"]),
                "expected_result": [],
            },
        ]

        for case in test_cases:
            with patch.object(
                graph_builder, "max_weight_shortest_path"
            ) as mock_max_weight_shortest_path:
                mock_max_weight_shortest_path.side_effect = case["mock_side_effect"]

                df_test = case.get("df", df)
                result = graph_builder.collect_path_total_weights(df_test)

                assert result == case["expected_result"]


def test_graph_skeleton():
    """Test the graph_skeleton method using actual data from mi_diff_df.csv and first_frame.pdb."""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "mi_diff_df.csv")
    mi_diff_df = pd.read_csv(file_path)
    pdb_file = file_path = os.path.join(current_dir, "first_frame.pdb")
    last_residue = 100
    graphdist = 5

    gb = GraphBuilder(
        pdb=pdb_file,
        last_residue=last_residue,
        mi_diff_df=mi_diff_df,
        graphdist=graphdist,
    )

    graph = gb.graph_skeleton()

    assert isinstance(graph, nx.Graph), "The result should be a NetworkX Graph object."

    assert len(graph.edges) > 0, "The graph should contain edges."
