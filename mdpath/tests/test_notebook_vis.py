import pytest
from unittest.mock import MagicMock, patch, mock_open
import re
import json
from mdpath.src.notebook_vis import (
    NotebookVisualization,
)  # Adjust import according to your module


def normalize_script(script):
    # Remove extra newlines and whitespace
    return re.sub(r"\s+", " ", script.strip())


def remove_radius_from_dicts(data_list):
    """Removes the 'radius' key from each dictionary in the list."""
    for item in data_list:
        if "radius" in item:
            del item["radius"]
    return data_list


@pytest.fixture
def mock_view():
    """Fixture to provide a mocked NGL view."""
    with patch("mdpath.src.notebook_vis.nv.show_file") as mock_show_file:
        mock_view = MagicMock()
        mock_show_file.return_value = mock_view
        yield mock_view


@pytest.fixture
def mock_json_data():
    """Fixture to provide mock JSON data."""
    return [
        {
            "clusterid": 1,
            "color": [1, 0, 0],
            "coord1": [0, 0, 0],
            "coord2": [1, 1, 1],
            "pathway_index": 0,
            "radius": 0.5,
        },
        {
            "clusterid": 1,
            "color": [0, 1, 0],
            "coord1": [1, 1, 1],
            "coord2": [2, 2, 2],
            "pathway_index": 1,
            "radius": 0.5,
        },
        {
            "clusterid": 2,
            "color": [0, 0, 1],
            "coord1": [2, 2, 2],
            "coord2": [3, 3, 3],
            "pathway_index": 2,
            "radius": 0.5,
        },
    ]


def test_initialization(mock_view, mock_json_data):
    """Test initialization of NotebookVisualization."""
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_json_data))):
        nbv = NotebookVisualization("dummy.pdb", "dummy.json")

    assert nbv.pdb_path == "dummy.pdb"
    assert nbv.json_path == "dummy.json"
    assert nbv.view == mock_view
    assert nbv.precomputed_data == mock_json_data


def test_load_precomputed_data(mock_json_data):
    """Test loading precomputed data from JSON."""
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_json_data))):
        # Mocking NGLView to bypass actual file handling
        mock_view = MagicMock()
        with patch("nglview.show_file", return_value=mock_view):
            nbv = NotebookVisualization("dummy.pdb", "dummy.json")


def test_generate_cluster_ngl_script(mock_view, mock_json_data):
    """Test generation of cluster-specific NGL script."""
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_json_data))):
        nbv = NotebookVisualization("dummy.pdb", "dummy.json")

        nbv.view._execute_js_code = MagicMock()

        nbv.generate_cluster_ngl_script()

        expected_script_1 = """
            var shape = new NGL.Shape('Cluster1');
            shape.addCylinder([0, 0, 0], [1, 1, 1], [1, 0, 0], 0.5);
            shape.addCylinder([1, 1, 1], [2, 2, 2], [0, 1, 0], 0.5);
            var shapeComp = this.stage.addComponentFromObject(shape);
            shapeComp.addRepresentation('buffer');
        """

        expected_script_2 = """
            var shape = new NGL.Shape('Cluster2');
            shape.addCylinder([2, 2, 2], [3, 3, 3], [0, 0, 1], 0.5);
            var shapeComp = this.stage.addComponentFromObject(shape);
            shapeComp.addRepresentation('buffer');
        """

        assert nbv.view._execute_js_code.call_count == 2
        calls = [call[0][0] for call in nbv.view._execute_js_code.call_args_list]

        assert normalize_script(expected_script_1) in [
            normalize_script(call) for call in calls
        ]
        assert normalize_script(expected_script_2) in [
            normalize_script(call) for call in calls
        ]
