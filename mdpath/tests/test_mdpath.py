"""
Unit and regression test for the mdpath package.
"""

# Import package, test suite, and other packages as needed
import sys
import numpy as np
import pandas as pd
import pytest
import networkx as nx
import mdpath
from multiprocessing import Pool
import mdpath.mdpath
import mdpath.mdpath
import mdpath.src
import mdpath.src.bootstrap
import mdpath.src.notebook_vis
import mdpath.src.structure
import mdpath.src.graph
import mdpath.src.cluster
import mdpath.src.mutual_information
import mdpath.src.visualization
import mdpath.src.notebook_vis
import tempfile
from unittest.mock import MagicMock, Mock, patch, call
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from tqdm import tqdm
from Bio import PDB
import os
import json
import nglview as nv
import importlib.util
import shutil
import mdpath.src.bootstrap
import subprocess
import mdpath.src.visualization
from io import StringIO


def test_mdpath_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mdpath" in sys.modules


def test_mdpath_wrong_input(tmp_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    mdpath_dir = os.path.join(project_root, "mdpath")

    top = os.path.join(script_dir, "test_topology.pdb")
    expected_message = "Both trajectory and topology files are required!"

    sys.path.insert(0, mdpath_dir)

    try:
        from mdpath.mdpath import main as mdpath_main
    except ImportError as e:
        raise ImportError(f"Error importing mdpath: {e}")

    original_cwd = os.getcwd()
    os.chdir(script_dir)

    original_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        sys.argv = ["mdpath", "-top", top]

        with pytest.raises(SystemExit) as exc_info:
            mdpath_main()

        output = sys.stdout.getvalue()
        assert expected_message in output

        assert exc_info.value.code != 0

    finally:
        sys.stdout = original_stdout
        os.chdir(original_cwd)


def test_mdpath_output_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    mdpath_dir = os.path.join(project_root, "mdpath")

    topology = os.path.join(script_dir, "test_topology.pdb")
    trajectory = os.path.join(script_dir, "test_trajectory.dcd")
    numpath = "10"
    bootstrap = "1"
    assert os.path.exists(topology), f"Topology file {topology} does not exist."
    assert os.path.exists(trajectory), f"Trajectory file {trajectory} does not exist."

    expected_files = [
        os.path.join(script_dir, "first_frame.pdb"),
        os.path.join(script_dir, "nmi_df.csv"),
        os.path.join(script_dir, "output.txt"),
        os.path.join(script_dir, "residue_coordinates.pkl"),
        os.path.join(script_dir, "cluster_pathways_dict.pkl"),
        os.path.join(script_dir, "clusters_paths.json"),
        os.path.join(script_dir, "precomputed_clusters_paths.json"),
        os.path.join(script_dir, "quick_precomputed_clusters_paths.json"),
        os.path.join(script_dir, "bootstrap/bootstrap_sample_0.txt"),
    ]

    sys.path.insert(0, mdpath_dir)

    try:
        from mdpath.mdpath import main as mdpath_main
    except ImportError as e:
        raise ImportError(f"Error importing mdpath: {e}")

    original_cwd = os.getcwd()
    os.chdir(script_dir)

    try:
        sys.argv = [
            "mdpath",
            "-top",
            topology,
            "-traj",
            trajectory,
            "-numpath",
            numpath,
            "-bs",
            bootstrap,
            "-lig",
            "272",
        ]

        mdpath_main()

        for file in expected_files:
            assert os.path.exists(file), f"Expected output file {file} not found."

    finally:
        for file in expected_files:
            if os.path.exists(file):
                os.remove(file)
        os.chdir(original_cwd)
