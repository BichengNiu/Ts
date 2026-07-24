"""Tests for dated intervention specifications and design matrices."""

import numpy as np
import pandas as pd
import pytest


def test_event_spec_defaults_to_period_mapping():
    from Ts.TsModels._intervention import EventSpec

    event = EventSpec(name="policy", dates=["2025-03-15"], kind="pulse")

    assert event.date_rule == "period"
    assert event.window is None
    assert event.reference is None
    assert event.name == "policy"
    assert event.dates == (pd.Timestamp("2025-03-15"),)


@pytest.mark.parametrize("kind", ["other", "", None])
def test_event_spec_rejects_unknown_kind(kind):
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="kind"):
        EventSpec(name="policy", dates=["2025-03-15"], kind=kind)


@pytest.mark.parametrize("name", ["", "   ", None])
def test_event_spec_rejects_empty_name(name):
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="name"):
        EventSpec(name=name, dates=["2025-03-15"], kind="pulse")


def test_event_spec_rejects_empty_or_duplicate_dates():
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="dates"):
        EventSpec(name="policy", dates=[], kind="pulse")
    with pytest.raises(ValueError, match="duplicate"):
        EventSpec(
            name="policy",
            dates=["2025-03-15", "2025-03-15"],
            kind="pulse",
        )


def test_event_spec_rejects_unknown_date_rule():
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="date_rule"):
        EventSpec(
            name="policy",
            dates=["2025-03-15"],
            kind="pulse",
            date_rule="nearest",
        )


def test_step_rejects_dynamic_window():
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="window.*pulse"):
        EventSpec(
            name="policy",
            dates=["2025-03-15"],
            kind="step",
            window=(-2, 4),
            reference=-1,
        )


@pytest.mark.parametrize(
    ("window", "reference"),
    [
        ((-2, 4), None),
        (None, -1),
        ((4, -2), 0),
        ((-2.0, 4), 0),
        ((False, 4), 0),
        ((-2, 4), 5),
    ],
)
def test_event_spec_rejects_invalid_window_reference(window, reference):
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="window|reference"):
        EventSpec(
            name="policy",
            dates=["2025-03-15"],
            kind="pulse",
            window=window,
            reference=reference,
        )


def test_reference_zero_is_valid_and_event_is_immutable():
    from dataclasses import FrozenInstanceError

    from Ts.TsModels._intervention import EventSpec

    event = EventSpec(
        name="policy",
        dates=["2025-03-15"],
        kind="pulse",
        window=(-2, 2),
        reference=0,
    )

    assert event.reference == 0
    with pytest.raises(FrozenInstanceError):
        event.name = "changed"


def test_period_maps_within_month_to_monthly_observation():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range("2025-01-01", periods=5, freq="MS")
    matrix, metadata = build_event_matrix(
        dates,
        [EventSpec("policy", ["2025-03-15"], "pulse")],
    )

    assert matrix["event__policy"].tolist() == [0, 0, 1, 0, 0]
    assert metadata["policy"].columns == ("event__policy",)
    assert metadata["policy"].mapped_positions == (2,)


@pytest.mark.parametrize(
    ("freq", "start", "event_date", "expected_position"),
    [
        ("ME", "2025-01-31", "2025-03-15", 2),
        ("D", "2025-01-01", "2025-01-03 12:00", 2),
        ("QS", "2025-01-01", "2025-05-10", 1),
    ],
)
def test_period_mapping_supports_common_regular_frequencies(
    freq,
    start,
    event_date,
    expected_position,
):
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range(start, periods=5, freq=freq)
    matrix, _ = build_event_matrix(
        dates,
        [EventSpec("policy", [event_date], "pulse")],
    )

    assert np.flatnonzero(matrix["event__policy"]).tolist() == [
        expected_position
    ]


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("next", 2),
        ("previous", 1),
    ],
)
def test_directional_date_rules(rule, expected):
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range("2025-01-01", periods=4, freq="MS")
    matrix, _ = build_event_matrix(
        dates,
        [
            EventSpec(
                "policy",
                ["2025-02-15"],
                "pulse",
                date_rule=rule,
            )
        ],
    )

    assert np.flatnonzero(matrix["event__policy"]).tolist() == [expected]


def test_exact_rejects_missing_date_inside_calendar():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.DatetimeIndex(["2025-01-01", "2025-01-03", "2025-01-04"])

    with pytest.raises(ValueError, match="exact.*2025-01-02"):
        build_event_matrix(
            dates,
            [
                EventSpec(
                    "policy",
                    ["2025-01-02"],
                    "pulse",
                    date_rule="exact",
                )
            ],
        )


def test_period_rejects_irregular_calendar():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.DatetimeIndex(
        ["2025-01-01", "2025-01-02", "2025-01-04"]
    )

    with pytest.raises(ValueError, match="frequency"):
        build_event_matrix(
            dates,
            [EventSpec("policy", ["2025-01-02"], "pulse")],
        )


def test_repeated_step_dates_create_cumulative_staircase():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range("2025-01-01", periods=6, freq="MS")
    matrix, _ = build_event_matrix(
        dates,
        [
            EventSpec(
                "policy",
                ["2025-02-10", "2025-04-20"],
                "step",
            )
        ],
    )

    assert matrix["event__policy"].tolist() == [0, 1, 1, 2, 2, 2]


def test_event_window_excludes_reference_and_counts_overlap():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range("2025-01-01", periods=7, freq="D")
    event = EventSpec(
        "announcement",
        ["2025-01-03", "2025-01-04"],
        "pulse",
        window=(-1, 2),
        reference=-1,
        date_rule="exact",
    )
    matrix, metadata = build_event_matrix(dates, [event])

    assert "event__announcement__m1" not in matrix
    assert matrix.loc["2025-01-04", "event__announcement__p1"] == 1
    assert matrix.loc["2025-01-04", "event__announcement__p0"] == 1
    assert metadata["announcement"].relative_periods == (0, 1, 2)


def test_step_before_target_slice_remains_active():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    calendar = pd.date_range("2025-01-01", periods=6, freq="MS")
    target = calendar[3:]
    matrix, _ = build_event_matrix(
        target,
        [EventSpec("policy", ["2025-02-15"], "step")],
        calendar=calendar,
    )

    assert matrix["event__policy"].tolist() == [1, 1, 1]


def test_event_outside_calendar_contributes_zero_until_extension():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    history = pd.date_range("2025-01-01", periods=4, freq="MS")
    event = EventSpec("policy", ["2025-06-15"], "pulse")

    history_matrix, _ = build_event_matrix(history, [event])
    extended = pd.date_range("2025-01-01", periods=6, freq="MS")
    future_matrix, _ = build_event_matrix(
        extended[-2:],
        [event],
        calendar=extended,
    )

    assert not history_matrix.to_numpy().any()
    assert future_matrix["event__policy"].tolist() == [0, 1]


def test_build_event_matrix_rejects_name_collisions():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range("2025-01-01", periods=4, freq="MS")
    event = EventSpec("policy", ["2025-02-01"], "pulse")

    with pytest.raises(ValueError, match="duplicate event name"):
        build_event_matrix(dates, [event, event])
    with pytest.raises(ValueError, match="collision"):
        build_event_matrix(
            dates,
            [event],
            reserved_names=["event__policy"],
        )


def test_build_event_matrix_requires_target_subset_and_matching_timezone():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    calendar = pd.date_range(
        "2025-01-01",
        periods=4,
        freq="D",
        tz="UTC",
    )
    event = EventSpec(
        "policy",
        [pd.Timestamp("2025-01-02", tz="Europe/London")],
        "pulse",
        date_rule="exact",
    )

    with pytest.raises(ValueError, match="timezone"):
        build_event_matrix(calendar, [event])
    with pytest.raises(ValueError, match="target dates"):
        build_event_matrix(
            pd.DatetimeIndex(["2025-01-05"], tz="UTC"),
            [],
            calendar=calendar,
        )


def _fitted_policy_model(seed=31):
    from Ts.TsModels import SARIMA
    from Ts.TsModels._intervention import EventSpec

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=120, freq="MS")
    step = np.zeros(120)
    step[30:] += 1
    step[80:] += 1
    other = np.zeros(120)
    other[[45, 95]] = 1
    y = 1.8 * step - 0.7 * other + rng.normal(scale=0.05, size=120)
    return SARIMA(
        pd.Series(y, index=dates),
        events=[
            EventSpec(
                "policy",
                dates[[30, 80]],
                "step",
                date_rule="exact",
            ),
            EventSpec(
                "other",
                dates[[45, 95]],
                "pulse",
                date_rule="exact",
            ),
        ],
        order=(0, 0, 0),
        trend="n",
    ).fit()


def test_policy_effect_is_event_design_contrast():
    from Ts.TsModels._intervention import PolicyEffectResult

    fitted = _fitted_policy_model()
    effect = fitted.policy_effect(
        events="policy",
        start="2010-01-01",
        end="2019-12-01",
        method="delta",
    )
    column = fitted.design_columns.index("event__policy")
    expected = (
        fitted._design_matrix[:, column]
        * fitted.params["event__policy"]
    )

    assert isinstance(effect, PolicyEffectResult)
    np.testing.assert_allclose(effect.effect.to_numpy(), expected)
    np.testing.assert_allclose(
        effect.factual_mean - effect.counterfactual_mean,
        effect.effect,
    )
    assert effect.cumulative_effect == pytest.approx(expected.sum())
    assert "因果" in effect.identification_note


def test_policy_effect_keeps_nonselected_events_in_both_paths():
    fitted = _fitted_policy_model()
    policy = fitted.policy_effect("policy", method="delta")
    other = fitted.policy_effect("other", method="delta")
    both = fitted.policy_effect(["policy", "other"], method="delta")

    np.testing.assert_allclose(both.effect, policy.effect + other.effect)
    assert policy.coefficients["event"].unique().tolist() == ["policy"]


def test_policy_effect_validates_event_selection_and_dates():
    fitted = _fitted_policy_model()

    with pytest.raises(ValueError, match="unknown event"):
        fitted.policy_effect("missing", method="delta")
    with pytest.raises(ValueError, match="duplicate"):
        fitted.policy_effect(["policy", "policy"], method="delta")
    with pytest.raises(ValueError, match="must not be empty"):
        fitted.policy_effect([], method="delta")
    with pytest.raises(ValueError, match="prediction date"):
        fitted.policy_effect(
            "policy",
            start="2010-01-15",
            method="delta",
        )


def test_policy_effect_summary_and_plot_are_self_contained():
    import matplotlib.pyplot as plt

    effect = _fitted_policy_model().policy_effect(
        "policy",
        method="delta",
    )

    summary = effect.summary()
    assert "policy" in summary
    assert "Cumulative effect" in summary
    assert "delta" in summary
    assert effect.identification_note in summary
    fig, axes = effect.plot()
    assert len(axes) == 2
    plt.close(fig)


def test_policy_effect_result_rejects_misaligned_paths():
    from Ts.TsModels._intervention import PolicyEffectResult

    dates = pd.date_range("2025-01-01", periods=2, freq="MS")
    series = pd.Series([1.0, 2.0], index=dates)

    with pytest.raises(ValueError, match="aligned"):
        PolicyEffectResult(
            coefficients=pd.DataFrame(),
            factual_mean=series,
            counterfactual_mean=series.iloc[:1],
            effect=series,
            lower=series,
            upper=series,
            cumulative_effect=3.0,
            cumulative_lower=2.0,
            cumulative_upper=4.0,
            pretrend_test=None,
            method="delta",
            identification_note="note",
        )


def _fitted_window_model(seed=47):
    from Ts.TsModels import SARIMA
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=100, freq="MS")
    event = EventSpec(
        "announcement",
        [dates[50]],
        "pulse",
        window=(-3, 2),
        reference=-1,
        date_rule="exact",
    )
    design, _ = build_event_matrix(dates, [event])
    coefficients = np.array([-0.1, 0.05, 1.2, 0.7, 0.3])
    y = design.to_numpy() @ coefficients
    y += rng.normal(scale=0.05, size=len(dates))
    return SARIMA(
        pd.Series(y, index=dates),
        events=[event],
        order=(0, 0, 0),
        trend="n",
    ).fit()


def test_delta_interval_uses_full_event_covariance():
    from Ts.TsModels._intervention import _contrast_standard_errors

    contrast = np.array([[1.0, 2.0]])
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])

    standard_error = _contrast_standard_errors(contrast, covariance)

    assert standard_error[0] == pytest.approx(np.sqrt(44.0))


def test_delta_cumulative_interval_uses_summed_contrast():
    from scipy.stats import norm

    from Ts.TsModels._intervention import _delta_intervals

    beta = np.array([1.0, 2.0])
    contrast = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])

    _, _, cumulative_lower, cumulative_upper = _delta_intervals(
        contrast,
        beta,
        covariance,
        alpha=0.05,
    )

    cumulative_contrast = contrast.sum(axis=0)
    cumulative = float(cumulative_contrast @ beta)
    cumulative_se = np.sqrt(
        cumulative_contrast @ covariance @ cumulative_contrast
    )
    margin = norm.ppf(0.975) * cumulative_se
    assert cumulative_lower == pytest.approx(cumulative - margin)
    assert cumulative_upper == pytest.approx(cumulative + margin)


def test_simulation_is_reproducible_and_keeps_joint_covariance():
    fitted = _fitted_policy_model()

    first = fitted.policy_effect(
        "policy",
        method="simulation",
        n_draws=500,
        seed=123,
    )
    second = fitted.policy_effect(
        "policy",
        method="simulation",
        n_draws=500,
        seed=123,
    )

    pd.testing.assert_series_equal(first.lower, second.lower)
    pd.testing.assert_series_equal(first.upper, second.upper)
    assert first.cumulative_lower == second.cumulative_lower
    assert first.cumulative_upper == second.cumulative_upper


def test_event_leads_receive_joint_wald_pretrend_test():
    effect = _fitted_window_model().policy_effect(
        "announcement",
        method="delta",
    )

    assert effect.pretrend_test["df"] == 2
    assert 0.0 <= effect.pretrend_test["p_value"] <= 1.0
    reference = effect.coefficients.query("relative_period == -1")
    assert len(reference) == 1
    assert reference.iloc[0]["coef"] == 0.0
    assert bool(reference.iloc[0]["fixed"])


def test_policy_effect_uses_exact_parameter_names():
    fitted = _fitted_window_model()
    fitted.params = dict(reversed(tuple(fitted.params.items())))

    effect = fitted.policy_effect("announcement", method="delta")

    expected = np.zeros(fitted.nobs)
    metadata = fitted._event_metadata["announcement"]
    for column in metadata.columns:
        position = fitted.design_columns.index(column)
        expected += fitted._design_matrix[:, position] * fitted.params[column]
    np.testing.assert_allclose(effect.effect, expected)


@pytest.mark.parametrize("n_draws", [0, -1, 1.5, True])
def test_policy_effect_rejects_invalid_draw_count(n_draws):
    with pytest.raises(ValueError, match="n_draws"):
        _fitted_policy_model().policy_effect(
            "policy",
            method="simulation",
            n_draws=n_draws,
        )


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_policy_effect_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        _fitted_policy_model().policy_effect(
            "policy",
            method="simulation",
            n_draws=10,
            seed=seed,
        )


def test_policy_effect_rejects_unknown_inference_method():
    with pytest.raises(ValueError, match="method"):
        _fitted_policy_model().policy_effect("policy", method="unknown")
