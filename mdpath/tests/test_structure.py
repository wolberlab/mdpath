import pytest
import numpy as np
import pandas as pd
import io
import MDAnalysis as mda
from pathlib import Path
from Bio import PDB
from io import StringIO
from unittest.mock import patch, MagicMock
from mdpath.src.structure import (
    StructureCalculations,
    DihedralAngles,
)  # Replace 'mymodule' with the actual module name


@pytest.fixture
def pdb_file():
    test_dir = Path(__file__).parent
    return test_dir / "test_topology.pdb"


@pytest.fixture
def dcd_file():
    test_dir = Path(__file__).parent
    return test_dir / "test_trajectory.dcd"


def test_structure_calculations_init(pdb_file):
    calc = StructureCalculations(pdb_file)
    assert isinstance(calc, StructureCalculations)
    assert calc.first_res_num == 1
    assert calc.last_res_num == 303
    assert calc.num_residues == 303  # There is only one residue, so the range is empty


def test_res_num_from_pdb(pdb_file):
    calc = StructureCalculations(pdb_file)
    first_res, last_res = calc.res_num_from_pdb()
    # Replace with actual expected results
    assert first_res == 1
    assert last_res == 303


def test_calculate_distance(pdb_file):
    calc = StructureCalculations(pdb_file)
    # Test valid distance calculation
    distance = calc.calculate_distance((1.0, 1.0, 1.0), (4.0, 5.0, 6.0))
    assert np.isclose(distance, 7.071)  # Example assertion with expected distance


def test_calculate_residue_surroundings(pdb_file):
    calc = StructureCalculations(pdb_file)
    df = calc.calculate_residue_suroundings(dist=2.0, mode="close")
    # Replace with actual assertions based on your expected DataFrame
    assert df is not None


def test_calculate_residue_surroundings_invalid_mode(pdb_file):
    calc = StructureCalculations(pdb_file)
    with pytest.raises(ValueError):
        calc.calculate_residue_suroundings(dist=2.0, mode="invalid")


# Mock DihedralAngles for parallel testing
@pytest.fixture
def mock_traj():
    class MockTraj:
        def __init__(self):
            self.residues = {1: MockResidue(), 2: MockResidue()}

    class MockResidue:
        def phi_selection(self):
            return "phi_selection_mock"

    return MockTraj()


def test_dihedral_angles_init(mock_traj):
    angles = DihedralAngles(mock_traj, 1, 2, 1)
    assert isinstance(angles, DihedralAngles)
    assert angles.first_res_num == 1
    assert angles.last_res_num == 2
    assert angles.num_residues == 1


@patch("mdpath.src.structure.Dihedral")
def test_calc_dihedral_angle_movement(mock_dihedral_class):
    """
    Test the `calc_dihedral_angle_movement` method of `DihedralAngles`.

    This test verifies that the method calculates the dihedral angle movement correctly
    given a mocked trajectory and dihedral angle results.
    """
    # TODO Implement this test
    # traj = mda.Universe(str(dcd_file), str(pdb_file))
    # angles = DihedralAngles(traj, 1, 2, 1)
    # res_id, movement = angles.calc_dihedral_angle_movement(1)
    # assert res_id == 1
    # assert np.array_equal(movement, [1, 1])
    pass


@patch("mdpath.src.structure.Pool")  # Mocking multiprocessing Pool
def test_calculate_dihedral_movement_parallel(mock_pool, mock_traj):
    angles = DihedralAngles(mock_traj, 1, 2, 1)
    mock_pool.return_value.__enter__.return_value.imap_unordered.return_value = [
        (1, np.array([1, 2]))
    ]
    df = angles.calculate_dihedral_movement_parallel(num_parallel_processes=1)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


class TestCalculateDihedralMovementsMultiTraj:
    """Tests for StructureCalculations.calculate_dihedral_movements_multi_traj."""

    @patch("mdpath.src.structure.DihedralAngles")
    @patch("mdpath.src.structure.mda.Universe")
    def test_basic_concatenation(self, mock_universe, mock_dihedral_cls, pdb_file):
        """Multiple trajectories produce a row-concatenated DataFrame."""
        df1 = pd.DataFrame({"Res 1": [0.1, 0.2], "Res 2": [0.3, 0.4]})
        df2 = pd.DataFrame({"Res 1": [0.5, 0.6, 0.7], "Res 2": [0.8, 0.9, 1.0]})

        mock_instance = MagicMock()
        mock_instance.calculate_dihedral_movement_parallel.side_effect = [df1, df2]
        mock_dihedral_cls.return_value = mock_instance

        calc = StructureCalculations(pdb_file)
        result = calc.calculate_dihedral_movements_multi_traj(
            ["traj1.dcd", "traj2.dcd"], str(pdb_file), 1
        )

        assert len(result) == 5  # 2 + 3 rows
        assert list(result.columns) == ["Res 1", "Res 2"]
        assert result.index.tolist() == [0, 1, 2, 3, 4]

    @patch("mdpath.src.structure.DihedralAngles")
    @patch("mdpath.src.structure.mda.Universe")
    def test_column_intersection(self, mock_universe, mock_dihedral_cls, pdb_file):
        """Residues missing in some replicas are dropped."""
        df1 = pd.DataFrame({"Res 1": [0.1], "Res 2": [0.2], "Res 3": [0.3]})
        df2 = pd.DataFrame({"Res 1": [0.4], "Res 3": [0.5]})  # Res 2 missing

        mock_instance = MagicMock()
        mock_instance.calculate_dihedral_movement_parallel.side_effect = [df1, df2]
        mock_dihedral_cls.return_value = mock_instance

        calc = StructureCalculations(pdb_file)
        result = calc.calculate_dihedral_movements_multi_traj(
            ["traj1.dcd", "traj2.dcd"], str(pdb_file), 1
        )

        assert "Res 2" not in result.columns
        assert list(result.columns) == ["Res 1", "Res 3"]
        assert len(result) == 2

    @patch("mdpath.src.structure.DihedralAngles")
    @patch("mdpath.src.structure.mda.Universe")
    def test_single_trajectory(self, mock_universe, mock_dihedral_cls, pdb_file):
        """Single trajectory returns its DataFrame unchanged."""
        df1 = pd.DataFrame({"Res 1": [0.1, 0.2], "Res 2": [0.3, 0.4]})

        mock_instance = MagicMock()
        mock_instance.calculate_dihedral_movement_parallel.return_value = df1
        mock_dihedral_cls.return_value = mock_instance

        calc = StructureCalculations(pdb_file)
        result = calc.calculate_dihedral_movements_multi_traj(
            ["traj1.dcd"], str(pdb_file), 1
        )

        assert len(result) == 2
        assert list(result.columns) == ["Res 1", "Res 2"]

    @patch("mdpath.src.structure.DihedralAngles")
    @patch("mdpath.src.structure.mda.Universe")
    def test_empty_trajectory_skipped(self, mock_universe, mock_dihedral_cls, pdb_file):
        """Empty DataFrames from failed trajectories are skipped."""
        df1 = pd.DataFrame()
        df2 = pd.DataFrame({"Res 1": [0.1, 0.2]})

        mock_instance = MagicMock()
        mock_instance.calculate_dihedral_movement_parallel.side_effect = [df1, df2]
        mock_dihedral_cls.return_value = mock_instance

        calc = StructureCalculations(pdb_file)
        result = calc.calculate_dihedral_movements_multi_traj(
            ["traj1.dcd", "traj2.dcd"], str(pdb_file), 1
        )

        assert len(result) == 2
        assert list(result.columns) == ["Res 1"]
