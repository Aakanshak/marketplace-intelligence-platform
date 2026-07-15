"""Reusable A/B testing utilities including CUPED implemented from scratch."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest


@dataclass
class TestResult:
    effect: float
    p_value: float
    ci_low: float
    ci_high: float


def required_sample_size(baseline_rate: float, relative_lift: float, power: float = .8, alpha: float = .05) -> int:
    treatment = baseline_rate * (1 + relative_lift)
    effect = proportion_effectsize(baseline_rate, treatment)
    return int(np.ceil(NormalIndPower().solve_power(abs(effect), power=power, alpha=alpha, ratio=1)))


def cuped_adjust(outcome: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return CUPED-adjusted outcome, theta, and variance reduction fraction.

    theta = Cov(Y, X) / Var(X). Centering X preserves the outcome mean while
    subtracting its predictable pre-period component from each observation.
    """
    y, x = np.asarray(outcome, float), np.asarray(covariate, float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("outcome and covariate must be same-length 1-D arrays")
    variance = np.var(x, ddof=1)
    if variance == 0:
        return y.copy(), 0.0, 0.0
    theta = np.cov(y, x, ddof=1)[0, 1] / variance
    adjusted = y - theta * (x - x.mean())
    reduction = 1 - np.var(adjusted, ddof=1) / np.var(y, ddof=1)
    return adjusted, float(theta), float(reduction)


def _mean_test(y: np.ndarray, treatment: np.ndarray) -> TestResult:
    a, b = y[treatment == 0], y[treatment == 1]
    effect = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = effect / se
    ci = (effect - 1.96 * se, effect + 1.96 * se)
    return TestResult(float(effect), float(2 * stats.norm.sf(abs(z))), float(ci[0]), float(ci[1]))


def simulate_experiment(n: int = 40_000, baseline_rate: float = .08, relative_lift: float = .05,
                        correlation: float = .5, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, n)
    latent = rng.normal(0, 1, n)
    pre_rate = np.clip(baseline_rate + .10 * latent + rng.normal(0, .04 * (1 - correlation + .05), n), .001, .65)
    probability = np.clip(baseline_rate + correlation * .12 * latent + treatment * baseline_rate * relative_lift, .001, .8)
    outcome = rng.binomial(1, probability).astype(float)
    adjusted, theta, reduction = cuped_adjust(outcome, pre_rate)
    counts = np.array([outcome[treatment == 1].sum(), outcome[treatment == 0].sum()])
    ns = np.array([(treatment == 1).sum(), (treatment == 0).sum()])
    z, naive_p = proportions_ztest(counts, ns)
    naive = _mean_test(outcome, treatment)
    naive.p_value = float(naive_p)
    cuped = _mean_test(adjusted, treatment)
    guardrails = {
        "refund_rate_delta": float(rng.normal(0, .0004)),
        "page_load_ms_delta": float(rng.normal(1, 3)),
        "seller_complaint_rate_delta": float(rng.normal(0, .0002)),
    }
    return {"treatment": treatment, "outcome": outcome, "pre_rate": pre_rate, "adjusted": adjusted,
            "theta": theta, "variance_reduction": reduction, "naive": naive, "cuped": cuped, "guardrails": guardrails}
