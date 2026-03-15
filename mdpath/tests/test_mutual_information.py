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

    # Add your expected values here
    expected_values = {
        ("Residue1", "Residue2"): 1.1102230246251565e-16,
        ("Residue1", "Residue3"): 0.006253825923384082,
        ("Residue2", "Residue3"): 0.006287366618772272,
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

    # Add your expected values here
    expected_values = {
        ("Residue1", "Residue2"): 0.6729976399990001,
        ("Residue1", "Residue3"): 0.6667438140756161,
        ("Residue2", "Residue3"): 0.6667102733802279,
    }
    for pair, expected in expected_values.items():
        actual = result.loc[result["Residue Pair"] == pair, "MI Difference"].values[0]
        assert np.isclose(
            actual, expected, atol=1e-5
        ), f"For pair {pair}, expected {expected} but got {actual}"
