import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import patch
import pickle
import os
import numpy as np
from mdpath.src.bootstrap import BootstrapAnalysis


@pytest.fixture
def load_pickle_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bs = os.path.join(current_dir, "bs_init.pkl")
    with open(bs, "rb") as file:
        df_all_residues = pickle.load(file)
        df_distant_residues = pickle.load(file)
        sorted_paths = pickle.load(file)
        num_bootstrap_samples = pickle.load(file)
        numpath = pickle.load(file)
        last_residue = pickle.load(file)
        graphdist = pickle.load(file)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb = os.path.join(current_dir, "first_frame.pdb")

    return (
        df_all_residues,
        df_distant_residues,
        sorted_paths,
        num_bootstrap_samples,
        numpath,
        pdb,
        last_residue,
        graphdist,
    )


def test_bootstrap_analysis_init(load_pickle_data):
    (
        df_all_residues,
        df_distant_residues,
        sorted_paths,
        num_bootstrap_samples,
        numpath,
        pdb,
        last_residue,
        graphdist,
    ) = load_pickle_data

    analysis = BootstrapAnalysis(
        df_all_residues=df_all_residues,
        df_distant_residues=df_distant_residues,
        sorted_paths=sorted_paths,
        num_bootstrap_samples=num_bootstrap_samples,
        numpath=numpath,
        pdb=pdb,
        last_residue=last_residue,
        graphdist=graphdist,
        num_bins=35,
    )

    assert_frame_equal(analysis.df_all_residues, df_all_residues)
    assert_frame_equal(analysis.df_distant_residues, df_distant_residues)

    assert analysis.sorted_paths == sorted_paths
    assert analysis.num_bootstrap_samples == num_bootstrap_samples
    assert analysis.numpath == numpath
    assert analysis.pdb == pdb
    assert analysis.last_residue == last_residue
    assert analysis.graphdist == graphdist
    assert analysis.num_bins == 35


@pytest.fixture
def load_dataframe():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mi_df = os.path.join(current_dir, "mi_diff_df.csv")
    df_dihedral = pd.read_csv(mi_df)
    return df_dihedral


def test_create_bootstrap_sample(load_dataframe, load_pickle_data):
    (
        df_all_residues,
        df_distant_residues,
        sorted_paths,
        num_bootstrap_samples,
        numpath,
        pdb,
        last_residue,
        graphdist,
    ) = load_pickle_data

    analysis = BootstrapAnalysis(
        df_all_residues=df_all_residues,
        df_distant_residues=df_distant_residues,
        sorted_paths=sorted_paths,
        num_bootstrap_samples=num_bootstrap_samples,
        numpath=numpath,
        pdb=pdb,
        last_residue=last_residue,
        graphdist=graphdist,
        num_bins=35,
    )

    bootstrap_sample = analysis.create_bootstrap_sample(load_dataframe)
    assert bootstrap_sample.shape == load_dataframe.shape
    assert list(bootstrap_sample.columns) == list(load_dataframe.columns)
    assert not bootstrap_sample.equals(
        load_dataframe
    ), "Bootstrap sample should not be identical to the original data"


@pytest.fixture
def setup_directories():
    if not os.path.exists("bootstrap"):
        os.makedirs("bootstrap")
    yield

    if os.path.exists("bootstrap/bootstrap_sample_1.txt"):
        os.remove("bootstrap/bootstrap_sample_1.txt")


def test_process_bootstrap_sample(load_dataframe, load_pickle_data, setup_directories):
    (
        df_all_residues,
        df_distant_residues,
        sorted_paths,
        num_bootstrap_samples,
        numpath,
        pdb,
        last_residue,
        graphdist,
    ) = load_pickle_data

    analysis = BootstrapAnalysis(
        df_all_residues=df_all_residues,
        df_distant_residues=df_distant_residues,
        sorted_paths=sorted_paths,
        num_bootstrap_samples=num_bootstrap_samples,
        numpath=numpath,
        pdb=pdb,
        last_residue=last_residue,
        graphdist=graphdist,
        num_bins=35,
    )

    pathways = [path for path, _ in analysis.sorted_paths[: analysis.numpath]]
    pathways_set = set(tuple(path) for path in pathways)

    common_count, bootstrap_pathways = analysis.process_bootstrap_sample(
        pathways_set, sample_num=1
    )

    assert isinstance(common_count, int), "common_count should be an integer"
    assert isinstance(bootstrap_pathways, list), "bootstrap_pathways should be a list"
    assert all(
        isinstance(path, list) for path in bootstrap_pathways
    ), "Each pathway should be a list"
    assert os.path.exists(
        "bootstrap/bootstrap_sample_1.txt"
    ), "Output file should be created"


def test_bootstrap_analysis(load_pickle_data, setup_directories):
    (
        df_all_residues,
        df_distant_residues,
        sorted_paths,
        num_bootstrap_samples,
        numpath,
        pdb,
        last_residue,
        graphdist,
    ) = load_pickle_data

    analysis = BootstrapAnalysis(
        df_all_residues=df_all_residues,
        df_distant_residues=df_distant_residues,
        sorted_paths=sorted_paths,
        num_bootstrap_samples=num_bootstrap_samples,
        numpath=numpath,
        pdb=pdb,
        last_residue=last_residue,
        graphdist=graphdist,
        num_bins=35,
    )

    common_counts, path_confidence_intervals = analysis.bootstrap_analysis()

    assert isinstance(
        common_counts, np.ndarray
    ), "common_counts should be a numpy array"
    assert (
        len(common_counts) == num_bootstrap_samples
    ), "Length of common_counts should match num_bootstrap_samples"
    assert isinstance(
        path_confidence_intervals, dict
    ), "path_confidence_intervals should be a dictionary"


def test_bootstrap_write(load_pickle_data, setup_directories, tmp_path):

    (
        df_all_residues,
        df_distant_residues,
        sorted_paths,
        num_bootstrap_samples,
        numpath,
        pdb,
        last_residue,
        graphdist,
    ) = load_pickle_data

    analysis = BootstrapAnalysis(
        df_all_residues=df_all_residues,
        df_distant_residues=df_distant_residues,
        sorted_paths=sorted_paths,
        num_bootstrap_samples=num_bootstrap_samples,
        numpath=numpath,
        pdb=pdb,
        last_residue=last_residue,
        graphdist=graphdist,
        num_bins=35,
    )

    analysis.common_counts, analysis.path_confidence_intervals = (
        analysis.bootstrap_analysis()
    )
    temp_file = tmp_path / "bootstrap_output.txt"

    analysis.bootstrap_write(str(temp_file))

    with open(temp_file, "r") as file:
        lines = file.readlines()

    for path, (mean, lower, upper) in analysis.path_confidence_intervals.items():
        path_str = " -> ".join(map(str, path))
        expected_line = f"{path_str}: Mean={mean}, 2.5%={lower}, 97.5%={upper}\n"
        assert (
            expected_line in lines
        ), f"Expected line '{expected_line.strip()}' not found in file"

    assert os.path.exists(temp_file), "Output file should be created"

    os.remove(temp_file)
