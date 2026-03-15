import pytest
import pandas as pd
from unittest.mock import patch
import pickle
import os
from mdpath.src.cluster import PatwayClustering


@pytest.fixture
def sample_data():
    df_basic = pd.DataFrame({"Residue1": [1, 2, 3, 4], "Residue2": [2, 3, 4, 5]})
    pathways = [[1, 2], [3, 4], [1, 4]]
    num_processes = 2
    return df_basic, pathways, num_processes


@pytest.fixture(scope="module")
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "cluster_data.pkl")
    with open(file_path, "rb") as file:
        df_close_res = pickle.load(file)
        top_pathways = pickle.load(file)
        num_parallel_processes = pickle.load(file)
    return df_close_res, top_pathways, num_parallel_processes


def test_initialization(load_data):
    df_close_res, top_pathways, num_parallel_processes = load_data

    clustering = PatwayClustering(df_close_res, top_pathways, num_parallel_processes)
    assert clustering.df.equals(df_close_res)
    assert clustering.pathways == top_pathways
    assert clustering.num_processes == num_parallel_processes


@patch("mdpath.src.cluster.PatwayClustering.calculate_overlap_parallel")
def test_calculate_overlap_parallel(mock_calculate_overlap_parallel, sample_data):
    df, pathways, num_processes = sample_data
    clustering = PatwayClustering(df, pathways, num_processes=num_processes)

    expected_data = [
        {"Pathway1": 0, "Pathway2": 1, "Overlap": 1},
        {"Pathway1": 1, "Pathway2": 0, "Overlap": 1},
        {"Pathway1": 0, "Pathway2": 2, "Overlap": 0},
        {"Pathway1": 2, "Pathway2": 0, "Overlap": 0},
        {"Pathway1": 1, "Pathway2": 2, "Overlap": 1},
        {"Pathway1": 2, "Pathway2": 1, "Overlap": 1},
    ]
    expected_df = pd.DataFrame(expected_data)
    mock_calculate_overlap_parallel.return_value = expected_df
    clustering.overlapp_df = clustering.calculate_overlap_parallel()
    pd.testing.assert_frame_equal(clustering.overlapp_df, expected_df)


@pytest.fixture(scope="module")
def load_test_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "cluster_data.pkl")
    with open(file_path, "rb") as file:
        df_close_res = pickle.load(file)
        top_pathways = pickle.load(file)
        num_parallel_processes = pickle.load(file)
    return df_close_res, top_pathways, num_parallel_processes


@pytest.fixture
def setup_clustering(load_test_data):
    df_close_res, top_pathways, num_parallel_processes = load_test_data

    return PatwayClustering(df_close_res, top_pathways, num_parallel_processes)


def test_pathways_cluster(setup_clustering):
    clustering = setup_clustering

    with patch("matplotlib.pyplot.savefig") as mock_savefig:
        clusters = clustering.pathways_cluster(n_top_clust=3)  #

    assert isinstance(clusters, dict)
    assert all(isinstance(k, int) and isinstance(v, list) for k, v in clusters.items())
    assert len(clusters) > 0

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "clusters.pkl")
    with open(file_path, "rb") as file:
        saved_clusters = pickle.load(file)

    assert clusters == saved_clusters

    mock_savefig.assert_called_once_with("clustered_paths.png")


@pytest.fixture(scope="module")
def load_cluster_pathways_dict():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "cluster_pathways_dict.pkl")
    with open(file_path, "rb") as pkl_file:
        return pickle.load(pkl_file)


def test_pathway_clusters_dictionary(setup_clustering, load_cluster_pathways_dict):
    clustering = setup_clustering

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "clusters.pkl")
    with open(file_path, "rb") as file:
        clusters = pickle.load(file)

    file_path = os.path.join(current_dir, "sorted_paths.pkl")
    with open(file_path, "rb") as file:
        sorted_paths = pickle.load(file)

    result_dict = clustering.pathway_clusters_dictionary(clusters, sorted_paths)

    assert result_dict == load_cluster_pathways_dict

    assert isinstance(result_dict, dict)
    for key, value in result_dict.items():
        assert isinstance(key, int)
        assert isinstance(value, list)
