import json
import numpy as np
import pandas as pd
import pytest

from mdpath.src.null_model import (
    BlockSplitNull,
    confidence_floor,
    _count_unique_groups,
    _observed_task,
)


def test_unique_group_count_ignores_ordering():
    # a group is a set of block indices: (1, 2) and (2, 1) are the same draw
    groups = [np.array([1, 2]), np.array([2, 1]), np.array([0, 3])]
    assert _count_unique_groups(groups) == 2


def _replicas():
    pattern = np.tile(np.linspace(-170, 170, 10, endpoint=False), 4)
    return [pd.DataFrame({"Res 1": pattern, "Res 2": pattern}) for _ in range(4)]


def _run(outdir):
    replicas = _replicas()
    return BlockSplitNull(
        replicas,
        replicas,
        block_lengths_ns=(10, 20),
        frames_per_ns=1,
        pairs=[(1, 2)],
        num_bins=10,
        draws=5,
        seed=12,
    ).run(outdir, primary_block_ns=10)


def test_perfectly_correlated_repeated_blocks_have_unit_null_confidence(tmp_path):
    edge = _run(tmp_path)["edge_null_confidence"].iloc[0]
    assert edge.cv_null_med < 1e-12
    assert edge.c_null_med > 1 - 1e-12


def test_remainder_is_discarded_without_overlapping_blocks():
    frame = pd.DataFrame({"Res 1": np.arange(23)})
    blocks, remainder = BlockSplitNull.split_blocks(frame, 5)
    assert len(blocks) * 5 <= len(frame)
    assert [len(block) for block in blocks] == [5, 5, 5, 5]
    assert remainder == 3
    assert blocks[-1]["Res 1"].tolist() == [15, 16, 17, 18, 19]


def test_outputs_are_deterministic_and_matched_design_is_equal(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    result = _run(first)
    _run(second)
    for name in (
        "edge_null_confidence",
        "null_by_source",
        "null_by_length",
        "matched_comparison",
    ):
        assert (first / f"{name}.csv").read_bytes() == (
            second / f"{name}.csv"
        ).read_bytes()
    matched = result["matched_comparison"]
    assert (matched.L == matched.L_obs).all()
    assert (matched.R_null == matched.R_obs).all()
    edge = result["edge_null_confidence"]
    assert (edge.c_obs.dropna() >= edge.c_min_R).all()
    assert (edge.c_null_med.dropna() >= edge.c_min_R).all()
    assert confidence_floor(4) == 1 / (1 + np.sqrt(3))


def test_corr_status_leaves_c_corr_experimental_unresolved(tmp_path):
    files = _run(tmp_path)
    edge = files["edge_null_confidence"]
    # the misleading 'clamped'->1 column is gone; the score is renamed experimental
    assert "clamped" not in edge.columns and "c_corr" not in edge.columns
    assert {"c_corr_experimental", "corr_status", "observed_below_reference"} <= set(
        edge.columns
    )
    assert set(edge.corr_status.unique()) <= {
        "undefined",
        "below_or_equal_reference",
        "above_reference",
    }
    resolved = edge.corr_status == "above_reference"
    assert edge.loc[resolved, "c_corr_experimental"].notna().all()
    assert edge.loc[~resolved, "c_corr_experimental"].isna().all()
    # machine-readable metadata is written alongside the CSVs
    assert (tmp_path / "run_metadata.json").exists()
    assert (tmp_path / "null_edge_by_source.csv").exists()


def test_nonfinite_and_equal_cv_are_unresolved_not_perfect(tmp_path):
    # A constant residue yields undefined NMI/CV; it must resolve to "undefined"
    # (c_corr NaN), never to a perfect score of 1.0.
    const = pd.DataFrame({"Res 1": np.ones(40), "Res 2": np.ones(40)})
    reps = [const.copy() for _ in range(4)]
    files = BlockSplitNull(
        reps,
        reps,
        block_lengths_ns=(10,),
        frames_per_ns=1,
        pairs=[(1, 2)],
        num_bins=10,
        draws=3,
        seed=1,
    ).run(tmp_path, primary_block_ns=10)
    edge = files["edge_null_confidence"]
    # constant series -> undefined MI/CV -> must NOT be reported as perfect (1.0)
    assert (edge.corr_status == "undefined").all()
    assert edge.c_corr_experimental.isna().all()
    assert not (edge.c_corr_experimental == 1.0).any()


def test_primary_point_is_invariant_to_added_sweep_length(tmp_path):
    # Adding an unrelated shorter block length must not change the primary (25 ns)
    # design point's reference values (design-point-specific RNG).
    reps = [
        pd.DataFrame(
            {
                "Res 1": np.sin(np.arange(400) + i),
                "Res 2": np.cos(np.arange(400) * 1.3 + i),
            }
        )
        for i in range(4)
    ]

    def primary(sweep, outdir):
        e = BlockSplitNull(
            reps,
            reps,
            block_lengths_ns=sweep,
            frames_per_ns=1,
            pairs=[(1, 2)],
            num_bins=8,
            draws=50,
            seed=7,
            n_jobs=1,
        ).run(outdir, primary_block_ns=25)["edge_null_confidence"]
        return e[["c_null_med", "rho"]].to_numpy()

    a = primary([25], tmp_path / "a")
    b = primary([5, 10, 25], tmp_path / "b")
    np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)


def _windowed_replicas(n=4, frames=120, seed=5):
    # distinct windows so the leading slice is not representative of the whole run
    rng = np.random.default_rng(seed)
    return [
        pd.DataFrame(
            {
                "Res 1": rng.uniform(-180, 180, frames),
                "Res 2": rng.uniform(-180, 180, frames),
            }
        )
        for _ in range(n)
    ]


def _matched(outdir, observed_window):
    reps = _windowed_replicas()
    return BlockSplitNull(
        reps,
        reps,
        block_lengths_ns=(30,),
        frames_per_ns=1,
        pairs=[(1, 2)],
        num_bins=8,
        draws=20,
        seed=4,
        observed_window=observed_window,
    ).run(outdir, primary_block_ns=30)["matched_comparison"]


def test_random_observed_windows_use_the_whole_replica(tmp_path):
    # "leading" evaluates one window per replica; "random" evaluates every window
    reps = _windowed_replicas()
    lengths = [(30.0, 30)]
    lead = _observed_task((reps[0], lengths, [(1, 2)], 8, "leading", (-180.0, 180.0)))
    rand = _observed_task((reps[0], lengths, [(1, 2)], 8, "random", (-180.0, 180.0)))
    assert lead[30.0].shape[0] == 1
    assert rand[30.0].shape[0] == len(reps[0]) // 30 == 4


def test_random_observed_windows_give_non_degenerate_uncertainty(tmp_path):
    # At R == number of replicas the replica set is fixed, so the legacy leading
    # window yields a single draw and zero observed uncertainty. Random windows
    # keep the draws genuinely different.
    lead = _matched(tmp_path / "lead", "leading")
    rand = _matched(tmp_path / "rand", "random")
    full_lead = lead[lead.R_null == lead.R_null.max()]
    full_rand = rand[rand.R_null == rand.R_null.max()]
    assert (full_lead.c_obs_matched_iqr == 0).all()
    assert (full_rand.c_obs_matched_iqr > 0).any()


def test_rho_interval_brackets_the_point_estimate(tmp_path):
    # needs replicas that actually differ, otherwise every CV is 0 and rho is NaN
    reps = _windowed_replicas()
    edge = BlockSplitNull(
        reps,
        reps,
        block_lengths_ns=(30,),
        frames_per_ns=1,
        pairs=[(1, 2)],
        num_bins=8,
        draws=20,
        seed=4,
    ).run(tmp_path, primary_block_ns=30)["edge_null_confidence"]
    band = edge[["rho", "rho_iqr_lo", "rho_iqr_hi"]].dropna()
    assert len(band)
    assert (band.rho_iqr_lo <= band.rho + 1e-12).all()
    assert (band.rho <= band.rho_iqr_hi + 1e-12).all()


def test_block_length_plateau_is_reported(tmp_path):
    _run(tmp_path)
    plateau = json.loads((tmp_path / "run_metadata.json").read_text())[
        "block_length_plateau"
    ]
    assert plateau["lengths_ns"] == [10.0, 20.0]
    assert plateau["plateaued"] in (True, False)
    assert plateau["final_rel_change"] >= 0


def test_observed_window_must_be_valid():
    reps = _windowed_replicas()
    with pytest.raises(ValueError, match="observed_window"):
        BlockSplitNull(
            reps,
            reps,
            block_lengths_ns=(30,),
            frames_per_ns=1,
            pairs=[(1, 2)],
            observed_window="first",
        )


def test_infeasible_lengths_are_dropped_with_warning():
    replicas = _replicas()  # 40 frames each at frames_per_ns=1
    with pytest.warns(UserWarning, match="infeasible"):
        null = BlockSplitNull(
            replicas,
            replicas,
            block_lengths_ns=(10, 20, 250),
            frames_per_ns=1,
            pairs=[(1, 2)],
            num_bins=10,
            draws=3,
            seed=1,
        )
    assert [length for length, _ in null.lengths] == [10.0, 20.0]  # 250 ns dropped


def test_infeasible_primary_length_raises(tmp_path):
    replicas = _replicas()
    with pytest.warns(UserWarning):
        null = BlockSplitNull(
            replicas,
            replicas,
            block_lengths_ns=(10, 250),
            frames_per_ns=1,
            pairs=[(1, 2)],
            num_bins=10,
            draws=3,
            seed=1,
        )
    with pytest.raises(ValueError, match="primary"):
        null.run(tmp_path, primary_block_ns=250)
