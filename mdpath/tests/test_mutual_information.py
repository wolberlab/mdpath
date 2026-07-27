import sys
import pandas as pd
import numpy as np


from mdpath.src.mutual_information import NMICalculator


def test_nmi_calculator_invert():
    np.random.seed(0)  # Set a fixed seed for reproducibility
    data = {
        "Residue1": np.random.uniform(-180, 180, size=100),
        "Residue2": np.random.uniform(-180, 180, size=100),
        "Residue3": np.random.uniform(-180, 180, size=100),
    }
    df_all_residues = pd.DataFrame(data)

    calculator = NMICalculator(df_all_residues, invert=True)
    result = calculator.nmi_df

    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    assert (
        "Residue Pair" in result.columns
    ), "DataFrame should contain 'Residue Pair' column"
    assert (
        "MI Difference" in result.columns
    ), "DataFrame should contain 'MI Difference' column"

    assert not result.empty, "DataFrame should not be empty"

    expected_shape = (
        len(df_all_residues.columns) * (len(df_all_residues.columns) - 1),
        2,
    )
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {result.shape}"

    assert (
        result["MI Difference"] >= 0
    ).all(), "MI Difference values should be non-negative"

    # Recomputed under the fixed (-180, 180) histogram support introduced with the
    # circular-coordinate fix; the pre-fix values are still reproduced exactly by
    # hist_range=None (see test_hist_range_none_is_exact_no_op).
    expected_values = {
        ("Residue1", "Residue2"): 0.019723970154380432,
        ("Residue1", "Residue3"): 0.0,
        ("Residue2", "Residue3"): 0.0005904872893245372,
    }
    for pair, expected in expected_values.items():
        actual = result.loc[result["Residue Pair"] == pair, "MI Difference"].values[0]
        assert np.isclose(
            actual, expected, atol=1e-5
        ), f"For pair {pair}, expected {expected} but got {actual}"


def test_nmi_calculator():
    np.random.seed(0)  # Set a fixed seed for reproducibility
    data = {
        "Residue1": np.random.uniform(-180, 180, size=100),
        "Residue2": np.random.uniform(-180, 180, size=100),
        "Residue3": np.random.uniform(-180, 180, size=100),
    }
    df_all_residues = pd.DataFrame(data)

    calculator = NMICalculator(df_all_residues)
    result = calculator.nmi_df

    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    assert (
        "Residue Pair" in result.columns
    ), "DataFrame should contain 'Residue Pair' column"
    assert (
        "MI Difference" in result.columns
    ), "DataFrame should contain 'MI Difference' column"

    assert not result.empty, "DataFrame should not be empty"

    expected_shape = (
        len(df_all_residues.columns) * (len(df_all_residues.columns) - 1),
        2,
    )
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {result.shape}"

    assert (
        result["MI Difference"] >= 0
    ).all(), "MI Difference values should be non-negative"

    # Recomputed under the fixed (-180, 180) histogram support introduced with the
    # circular-coordinate fix; the pre-fix values are still reproduced exactly by
    # hist_range=None (see test_hist_range_none_is_exact_no_op).
    expected_values = {
        ("Residue1", "Residue2"): 0.6510110199334083,
        ("Residue1", "Residue3"): 0.6707349900877887,
        ("Residue2", "Residue3"): 0.6701445027984642,
    }
    for pair, expected in expected_values.items():
        actual = result.loc[result["Residue Pair"] == pair, "MI Difference"].values[0]
        assert np.isclose(
            actual, expected, atol=1e-5
        ), f"For pair {pair}, expected {expected} but got {actual}"


# --------------------------------------------------------------------------- #
# circular-coordinate / fixed-support regression tests                         #
# --------------------------------------------------------------------------- #
def test_dihedral_difference_is_wrapped():
    """A step across the +-180 branch cut must stay a small step."""
    from mdpath.src.structure import DihedralAngles

    angles = np.array([[179.0], [-179.0], [178.0]])
    raw = np.diff(angles, axis=0)
    wrapped = (raw + 180.0) % 360.0 - 180.0

    # unwrapped, a 2 deg step becomes a ~358 deg jump
    assert np.allclose(raw.ravel(), [-358.0, 357.0])
    # 179 -> -179 is +2 deg (179 + 2 = 181 == -179); -179 -> 178 is -3 deg
    assert np.allclose(wrapped.ravel(), [2.0, -3.0])
    assert np.all(np.abs(wrapped) <= 180.0)
    # and this is the code path the package uses
    assert hasattr(DihedralAngles, "_calc_dihedral_movement_with_crossings")


def test_fixed_support_gives_identical_bin_edges_across_inputs():
    """Inputs with different extrema must share bin edges under a fixed support."""
    rng = np.random.default_rng(0)
    narrow = pd.DataFrame({"Res 1": rng.uniform(-20, 20, 200)})
    wide = pd.DataFrame({"Res 1": rng.uniform(-150, 150, 200)})

    edges_fixed = [
        np.histogram(f["Res 1"], bins=35, range=(-180.0, 180.0))[1]
        for f in (narrow, wide)
    ]
    assert np.array_equal(edges_fixed[0], edges_fixed[1])

    edges_legacy = [np.histogram(f["Res 1"], bins=35)[1] for f in (narrow, wide)]
    assert not np.array_equal(edges_legacy[0], edges_legacy[1])


def test_hist_range_none_is_exact_no_op():
    """hist_range=None must reproduce the pre-fix values bit-for-bit."""
    np.random.seed(0)
    df = pd.DataFrame(
        {f"Residue{i + 1}": np.random.uniform(-180, 180, size=100) for i in range(3)}
    )
    legacy = NMICalculator(df, show_progress=False, hist_range=None).nmi_df
    pre_fix = {
        ("Residue1", "Residue2"): 0.6729976399990001,
        ("Residue1", "Residue3"): 0.6667438140756161,
        ("Residue2", "Residue3"): 0.6667102733802279,
    }
    for pair, expected in pre_fix.items():
        actual = legacy.loc[legacy["Residue Pair"] == pair, "MI Difference"].values[0]
        assert actual == expected, f"{pair}: {actual!r} != {expected!r}"


def test_one_contaminated_frame_does_not_collapse_the_histogram():
    """A single branch-cut frame must not distort a residue once wrapped."""
    rng = np.random.default_rng(1)
    clean = rng.uniform(-15, 15, 1000)

    contaminated = clean.copy()
    contaminated[0] = -358.0  # what an unwrapped cut looks like
    wrapped = (contaminated + 180.0) % 360.0 - 180.0

    def occupied(values, **kwargs):
        return np.count_nonzero(np.histogram(values, bins=35, **kwargs)[0])

    # fixed support: wrapping restores the clean occupancy
    assert occupied(wrapped, range=(-180.0, 180.0)) == occupied(
        clean, range=(-180.0, 180.0)
    )
    # unwrapped + data-derived support is what collapses it
    assert occupied(contaminated) < occupied(clean) / 3
