import matplotlib

matplotlib.use("Agg")

import sys
import types

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import TwoSlopeNorm, Normalize

from mdpath.src.path_confidence_viz import ConfidencePathVisualizer as CPV


# --------------------------------------------------------------------------- #
# synthetic, self-contained inputs                                            #
# --------------------------------------------------------------------------- #
def _residue_coordinates():
    """Six CA points spread in 3D (float32, like MDPath's real output)."""
    return {
        1: [np.array([0.0, 0.0, 0.0], dtype=np.float32)],
        2: [np.array([10.0, 0.0, 0.0], dtype=np.float32)],
        3: [np.array([20.0, 5.0, 0.0], dtype=np.float32)],
        4: [np.array([30.0, 5.0, 5.0], dtype=np.float32)],
        5: [np.array([40.0, 10.0, 5.0], dtype=np.float32)],
        6: [np.array([50.0, 10.0, 10.0], dtype=np.float32)],
    }


def _top_pathways():
    # edges used: (1,2)x2, (2,3)x3, (3,4)x1, (3,5)x1, (3,6)x1  -> max usage = 3
    return [[1, 2, 3, 4], [1, 2, 3, 5], [2, 3, 6]]


def _edge_confidence_df():
    # some rows reversed (Residue1 > Residue2) to exercise canonicalization
    return pd.DataFrame(
        {
            "Residue1": [2, 2, 4, 3, 6],
            "Residue2": [1, 3, 3, 5, 3],
            "confidence": [0.95, 0.90, 0.60, 0.75, 0.55],
        }
    )


# --------------------------------------------------------------------------- #
# mock ChimeraX so the generated script can be executed as ChimeraX would     #
# --------------------------------------------------------------------------- #
class _FakeResidue:
    def __init__(self):
        self.name = None
        self.number = None


class _FakeMarker:
    def __init__(self, xyz, rgba, radius):
        self.coord, self.rgba, self.radius = xyz, rgba, radius
        self.residue = _FakeResidue()


class _FakeModels:
    def __init__(self):
        self.added = []

    def add(self, models):
        for m in models:
            m.id = (1,)
        self.added.extend(models)


class _FakeSession:
    def __init__(self):
        self.models = _FakeModels()


@pytest.fixture
def fake_chimerax(monkeypatch):
    """Install fake chimerax.* modules; return a recorder dict."""
    rec = {"cmds": [], "markers": [], "links": 0}

    def run(session, cmd):
        rec["cmds"].append(cmd)

    class _MarkerSet:
        def __init__(self, session, name):
            self.name = name
            self.id = None

        def create_marker(self, xyz, rgba, radius):
            m = _FakeMarker(xyz, rgba, radius)
            rec["markers"].append(m)
            return m

    def create_link(a, b, rgba=None, radius=None):
        rec["links"] += 1

    class _Atom:
        @staticmethod
        def register_attr(*args, **kwargs):
            pass

    mods = {n: types.ModuleType(n) for n in (
        "chimerax", "chimerax.core", "chimerax.core.commands",
        "chimerax.markers", "chimerax.markers.markers", "chimerax.atomic",
    )}
    mods["chimerax.core.commands"].run = run
    mods["chimerax.core"].commands = mods["chimerax.core.commands"]
    mods["chimerax"].core = mods["chimerax.core"]
    mods["chimerax.markers"].MarkerSet = _MarkerSet
    mods["chimerax.markers"].create_link = create_link
    mods["chimerax.markers.markers"].create_link = create_link
    mods["chimerax.atomic"].Atom = _Atom
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return rec


def _run_generated(path):
    ns = {"session": _FakeSession()}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return ns


# --------------------------------------------------------------------------- #
# confidence lookup + edge -> node lifting                                     #
# --------------------------------------------------------------------------- #
def test_conf_lookup_canonicalizes_pairs():
    lut = CPV.conf_lookup_from_df(_edge_confidence_df())
    assert lut[(1, 2)] == 0.95          # stored as (2, 1)
    assert lut[(3, 4)] == 0.60          # stored as (4, 3)
    assert lut[(3, 6)] == 0.55          # stored as (6, 3)
    assert (2, 3) in lut and (3, 5) in lut


def test_edge_and_node_confidence_lifting():
    lut = CPV.conf_lookup_from_df(_edge_confidence_df())
    res, edge, node = CPV.edge_and_node_confidence([1, 2, 3, 4], lut)
    assert res == [1, 2, 3, 4]
    np.testing.assert_allclose(edge, [0.95, 0.90, 0.60])
    # endpoints take their single edge; interior nodes average the two
    np.testing.assert_allclose(node, [0.95, (0.95 + 0.90) / 2, (0.90 + 0.60) / 2, 0.60])


def test_edge_and_node_confidence_handles_missing_edges():
    lut = CPV.conf_lookup_from_df(_edge_confidence_df())
    # (1,4) and (4,1) are not present -> NaN edge; node lifting is NaN-tolerant
    res, edge, node = CPV.edge_and_node_confidence([1, 4, 5], lut)
    assert np.isnan(edge[0])             # (1,4) missing
    assert edge[1] == 0.75 or np.isnan(edge[1])  # (4,5) missing -> NaN
    # node[1] averages the finite neighbours only (here both missing -> NaN)
    assert np.isnan(node[0])


# --------------------------------------------------------------------------- #
# auto colour ranging (the user's question)                                    #
# --------------------------------------------------------------------------- #
def test_colors_auto_range_from_input():
    """The colour scale is derived from the input confidence distribution."""
    low = [0.20, 0.30, 0.40, 0.50, 0.60]
    high = [0.80, 0.85, 0.90, 0.95, 0.97]
    norm_low = CPV.build_norm(low)
    norm_high = CPV.build_norm(high)

    assert isinstance(norm_low, TwoSlopeNorm)
    # centre is the median of each input
    assert norm_low.vcenter == pytest.approx(np.median(low))
    assert norm_high.vcenter == pytest.approx(np.median(high))
    # bounds are the 2nd / 98th percentiles of each input
    assert norm_low.vmin == pytest.approx(np.quantile(low, 0.02))
    assert norm_low.vmax == pytest.approx(np.quantile(low, 0.98))
    # the two scales adapt: ranges are different and disjoint, not hardcoded
    assert norm_low.vmax < norm_high.vmin


def test_build_norm_respects_explicit_overrides():
    norm = CPV.build_norm([0.1, 0.2, 0.9], vmin=0.0, vcenter=0.5, vmax=1.0)
    assert (norm.vmin, norm.vcenter, norm.vmax) == (0.0, 0.5, 1.0)


def test_build_norm_degenerate_inputs():
    assert isinstance(CPV.build_norm([]), Normalize)        # empty -> plain 0..1
    norm_const = CPV.build_norm([0.5, 0.5, 0.5])            # all equal -> no crash
    assert norm_const.vmin < norm_const.vmax


# --------------------------------------------------------------------------- #
# colour mapping direction + NaN                                               #
# --------------------------------------------------------------------------- #
def test_rgb_blue_high_red_low_and_nan():
    cmap = matplotlib.colormaps["RdBu"]
    norm = CPV.build_norm([0.0, 0.5, 1.0], vmin=0.0, vcenter=0.5, vmax=1.0)
    hi = CPV._rgb(0.95, norm, cmap)
    lo = CPV._rgb(0.05, norm, cmap)
    assert hi[2] > hi[0]                # high confidence -> blue dominates
    assert lo[0] > lo[2]               # low confidence  -> red dominates
    assert CPV._rgb(float("nan"), norm, cmap) == CPV.NAN_RGB


# --------------------------------------------------------------------------- #
# auto radius scaling                                                          #
# --------------------------------------------------------------------------- #
def test_auto_radius_scales_with_structure_size():
    coords = _residue_coordinates()
    arr = np.array([c[0] for c in coords.values()], float)
    diag = np.linalg.norm(arr.max(0) - arr.min(0))
    rmin, rmax = CPV.auto_radius_range(coords)
    assert rmax == pytest.approx(diag * CPV.AUTOSCALE_FACTOR)
    assert rmin == pytest.approx(rmax * CPV.AUTOSCALE_RATIO)
    assert 0 < rmin < rmax
    # scale multiplier roughly doubles the radius (within abs clamp)
    _, rmax2 = CPV.auto_radius_range(coords, scale=2.0)
    assert rmax2 == pytest.approx(min(rmax * 2.0, CPV.RADIUS_ABS_BOUNDS[1]))


def test_auto_radius_clamps_tiny_and_huge():
    tiny = {1: [np.zeros(3)], 2: [np.array([1.0, 0.0, 0.0])]}
    _, rmax_tiny = CPV.auto_radius_range(tiny)
    assert rmax_tiny == pytest.approx(0.4)                  # lower clamp
    huge = {1: [np.zeros(3)], 2: [np.array([5000.0, 0.0, 0.0])]}
    _, rmax_huge = CPV.auto_radius_range(huge)
    assert rmax_huge == pytest.approx(CPV.RADIUS_ABS_BOUNDS[1])  # upper clamp


# --------------------------------------------------------------------------- #
# spline: shape, in-range, no overshoot                                        #
# --------------------------------------------------------------------------- #
def test_spline_in_range_and_shape_preserving():
    coords = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 1]], float)
    node_conf = np.array([0.6, 0.9, 0.7, 0.95])
    node_radius = np.array([0.3, 1.0, 0.5, 0.3])
    pts, cf, rf = CPV._spline(coords, node_conf, node_radius, 50, 0.3, 1.0)
    assert pts.shape == (50, 3) and cf.shape == (50,) and rf.shape == (50,)
    # spline passes through the endpoints
    np.testing.assert_allclose(pts[0], coords[0], atol=1e-6)
    np.testing.assert_allclose(pts[-1], coords[-1], atol=1e-6)
    # PCHIP does not overshoot beyond the node value range (stable colours)
    assert cf.min() >= node_conf.min() - 1e-9
    assert cf.max() <= node_conf.max() + 1e-9
    # radius stays within the requested bounds
    assert rf.min() >= 0.3 - 1e-9 and rf.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# write_chimerax_script: summary, valid output, auto behaviour                 #
# --------------------------------------------------------------------------- #
def test_write_chimerax_script_summary_and_valid_python(tmp_path):
    out = tmp_path / "viewer.py"
    summary = CPV.write_chimerax_script(
        _top_pathways(), _residue_coordinates(), _edge_confidence_df(),
        pdb_file="does_not_exist.pdb", out_path=str(out),
        top_n=3, embed_structure=False,
    )
    assert out.exists()
    assert summary["paths"] == 3
    assert summary["sample_points"] == 3 * 70
    assert summary["radius_min"] < summary["radius_max"]
    assert summary["vmin"] <= summary["vcenter"] <= summary["vmax"]
    compile(out.read_text(), str(out), "exec")    # generated file is valid Python


def test_write_chimerax_script_color_range_follows_input(tmp_path):
    """Two inputs with different confidence ranges -> different colour scales."""
    coords, paths = _residue_coordinates(), [[1, 2, 3, 4]]
    df_lo = pd.DataFrame({"Residue1": [1, 2, 3], "Residue2": [2, 3, 4],
                          "confidence": [0.30, 0.40, 0.50]})
    df_hi = pd.DataFrame({"Residue1": [1, 2, 3], "Residue2": [2, 3, 4],
                          "confidence": [0.85, 0.90, 0.95]})
    s_lo = CPV.write_chimerax_script(paths, coords, df_lo,
                                     out_path=str(tmp_path / "lo.py"),
                                     top_n=1, embed_structure=False)
    s_hi = CPV.write_chimerax_script(paths, coords, df_hi,
                                     out_path=str(tmp_path / "hi.py"),
                                     top_n=1, embed_structure=False)
    assert s_hi["vmin"] > s_lo["vmax"]            # scale tracked the input range


def test_write_chimerax_script_radius_override(tmp_path):
    s = CPV.write_chimerax_script(
        _top_pathways(), _residue_coordinates(), _edge_confidence_df(),
        out_path=str(tmp_path / "v.py"), top_n=3, embed_structure=False,
        radius_min=0.5, radius_max=2.0,
    )
    assert (s["radius_min"], s["radius_max"]) == (0.5, 2.0)


def test_write_chimerax_script_raises_without_coordinates(tmp_path):
    df = _edge_confidence_df()
    with pytest.raises(ValueError):
        CPV.write_chimerax_script([[100, 101]], _residue_coordinates(), df,
                                  out_path=str(tmp_path / "v.py"),
                                  embed_structure=False)


# --------------------------------------------------------------------------- #
# generated script actually builds the marker model (mock ChimeraX)            #
# --------------------------------------------------------------------------- #
def test_generated_script_builds_marker_model(tmp_path, fake_chimerax):
    out = tmp_path / "viewer.py"
    CPV.write_chimerax_script(
        _top_pathways(), _residue_coordinates(), _edge_confidence_df(),
        out_path=str(out), top_n=3, embed_structure=False,
    )
    _run_generated(out)
    markers = fake_chimerax["markers"]
    assert len(markers) == 3 * 70                       # samples per path
    assert fake_chimerax["links"] == 3 * 69            # links between samples
    # hover encoding: residue name + percent number
    assert {m.residue.name for m in markers} == {"CONF"}
    assert all(0 <= m.residue.number <= 100 for m in markers)
    # exact value stored as B-factor; colours are valid RGBA bytes
    assert all(0.0 <= m.bfactor <= 1.0 for m in markers)
    assert all(len(m.rgba) == 4 and all(0 <= c <= 255 for c in m.rgba) for m in markers)
    # scene contains a colour-key legend and a framing 'view'
    assert any(c.startswith("key ") for c in fake_chimerax["cmds"])
    assert "view" in fake_chimerax["cmds"]


def test_generated_script_marks_undefined_confidence_gray(tmp_path, fake_chimerax):
    coords = {1: [np.array([0.0, 0.0, 0.0])], 2: [np.array([12.0, 0.0, 0.0])]}
    df = pd.DataFrame({"Residue1": [7], "Residue2": [8], "confidence": [0.9]})  # no (1,2)
    out = tmp_path / "viewer.py"
    CPV.write_chimerax_script([[1, 2]], coords, df, out_path=str(out),
                              top_n=1, embed_structure=False)
    _run_generated(out)
    gray = tuple(int(round(0.6 * 255)) for _ in range(3)) + (255,)
    markers = fake_chimerax["markers"]
    assert markers and all(tuple(m.rgba) == gray for m in markers)
    assert all(m.residue.number is None for m in markers)   # not encoded when undefined


# --------------------------------------------------------------------------- #
# the standalone CLI                                                           #
# --------------------------------------------------------------------------- #
def test_confidence_spline_cli(tmp_path, monkeypatch):
    import pickle

    paths_pkl = tmp_path / "top_pathways.pkl"
    coords_pkl = tmp_path / "residue_coordinates.pkl"
    conf_csv = tmp_path / "edge_confidence.csv"
    with open(paths_pkl, "wb") as f:
        pickle.dump(_top_pathways(), f)
    with open(coords_pkl, "wb") as f:
        pickle.dump(_residue_coordinates(), f)
    _edge_confidence_df().to_csv(conf_csv, index=False)
    out = tmp_path / "cli_viewer.py"

    monkeypatch.setattr(sys, "argv", [
        "mdpath_confidence_spline",
        "-paths", str(paths_pkl), "-coords", str(coords_pkl),
        "-conf", str(conf_csv), "-o", str(out),
        "-top", "3", "-pdb", "no_such_structure.pdb",
    ])
    from mdpath.mdpath_tools import confidence_spline

    with pytest.raises(SystemExit) as exc:
        confidence_spline()
    assert exc.value.code == 0
    assert out.exists()
    compile(out.read_text(), str(out), "exec")
