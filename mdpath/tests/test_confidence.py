import numpy as np
import pandas as pd
import pytest

from mdpath.src.confidence import EdgeConfidenceCalculator
from mdpath.src.edge_utils import canonical_pair, residue_id


def _nmi(value):
    return pd.DataFrame(
        {"Residue Pair": [("Res 1", "Res 2")], "MI Difference": [value]}
    )


def test_confidence_reports_finite_replica_floor():
    calculator = object.__new__(EdgeConfidenceCalculator)
    calculator.ddof = 0
    calculator.per_replica_nmi_dfs = [_nmi(1.0), _nmi(0.0), _nmi(0.0)]

    row = calculator.build_edge_confidence_df().iloc[0]

    assert row.MI_cv == pytest.approx(np.sqrt(2))
    assert row.confidence == pytest.approx(1 / (1 + np.sqrt(2)))
    # the maximally-inconsistent edge sits exactly on the finite-replica floor
    assert row.c_min == pytest.approx(row.confidence)
    assert row.c_norm == pytest.approx(0.0)
    assert row.ddof == 0


@pytest.mark.parametrize("ddof", [0, 1])
def test_c_norm_is_clipped_to_unit_interval(ddof):
    # three perfectly-consistent replicas -> confidence 1 -> c_norm 1;
    # a maximally-inconsistent edge -> c_norm 0. Both must stay in [0, 1].
    calculator = object.__new__(EdgeConfidenceCalculator)
    calculator.ddof = ddof
    calculator.per_replica_nmi_dfs = [_nmi(2.0), _nmi(2.0), _nmi(2.0)]
    row = calculator.build_edge_confidence_df().iloc[0]
    assert row.confidence == pytest.approx(1.0)
    assert 0.0 <= row.c_norm <= 1.0
    assert row.c_norm == pytest.approx(1.0)


def test_canonical_pair_and_residue_id():
    assert residue_id("Res 42") == 42
    assert residue_id(7) == 7
    assert canonical_pair(("Res 5", "Res 2")) == (2, 5)
    assert canonical_pair((9, 3)) == (3, 9)
