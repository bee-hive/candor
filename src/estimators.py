"""Off-policy evaluation estimators.

Each estimator takes a single dataset (1D arrays of length N) and returns a
single scalar estimate. The caller is responsible for any Monte-Carlo
replication (looping over random seeds).

The estimators are environment-agnostic: they consume callable reward models
and policies, so the same code works whether the env is a 2-state tabular
bandit, HeartSteps with linear-regression reward models over phi(s, a)
features, or the sepsis simulator with a larger discrete state/action space.

Conventions
-----------
- ``x``           : array of state representations, shape ``(N,)`` or ``(N, d)``.
                    Each ``x[i]`` is whatever the callables expect as a
                    "state" argument.
- ``a``           : array of factual actions, shape ``(N,)``, integer-valued.
- ``r``           : array of factual rewards, shape ``(N,)``.
- ``c``           : array of counterfactual-availability flags, shape ``(N,)``,
                    values in {0, 1}.
- ``rc``          : array of counterfactual rewards, shape ``(N,)``;
                    ``np.nan`` where ``c == 0``.
- ``cf_action``   : array of counterfactual actions, shape ``(N,)``; the action
                    the counterfactual annotation is for. For binary actions
                    this is typically ``1 - a``. Ignored where ``c == 0``.
- ``r_hat``       : callable ``(x_i, action) -> float``. Reward model fit on
                    factual data only.
- ``r_hat_plus``  : callable ``(x_i, action) -> float``. Reward model fit on
                    factual + counterfactual data.
- ``policy_b``    : callable ``(x_i, action) -> probability``.
- ``policy_e``    : callable ``(x_i, action) -> probability``.
- ``policy_b_plus``: callable ``(x_i, action) -> probability``. Augmented
                    behavior policy used by the counterfactual-IS estimators.
- ``weight_factual``: float in ``[0, 1]``. Weight assigned to the factual
                    sample when a counterfactual annotation is available; the
                    counterfactual sample receives ``1 - weight_factual``.

Misspecification of the reward model is handled outside these estimators by
passing in a corrupted ``r_hat`` (e.g., one fit on noisy state labels).
"""
import numpy as np


def standard_is(x, a, r, policy_b, policy_e):
    total = 0.0
    for xi, ai, ri in zip(x, a, r):
        total += (policy_e(xi, ai) / policy_b(xi, ai)) * ri
    return total / len(x)


def direct_method(x, r_hat, policy_e, n_actions):
    total = 0.0
    for xi in x:
        for act in range(n_actions):
            total += policy_e(xi, act) * r_hat(xi, act)
    return total / len(x)


def dm_is(x, a, r, r_hat, policy_b, policy_e, n_actions):
    total = 0.0
    for xi, ai, ri in zip(x, a, r):
        dm = sum(policy_e(xi, act) * r_hat(xi, act) for act in range(n_actions))
        correction = (policy_e(xi, ai) / policy_b(xi, ai)) * (ri - r_hat(xi, ai))
        if np.isnan(correction):
            correction = 0.0
        total += dm + correction
    return total / len(x)


def cis(x, a, r, c, rc, cf_action, policy_e, policy_b_plus, weight_factual):
    weight_cf = 1.0 - weight_factual
    total = 0.0
    for xi, ai, ri, ci, rci, cfi in zip(x, a, r, c, rc, cf_action):
        if ci == 1:
            w_f, w_cf = weight_factual, weight_cf
        else:
            w_f, w_cf = 1.0, 0.0
        factual = w_f * (policy_e(xi, ai) / policy_b_plus(xi, ai)) * ri
        cf = 0.0
        if ci == 1 and not np.isnan(rci):
            cf = w_cf * (policy_e(xi, cfi) / policy_b_plus(xi, cfi)) * rci
        total += factual + cf
    return total / len(x)


def dm_cis(x, a, r, c, rc, cf_action, r_hat, policy_e, policy_b_plus,
           weight_factual, n_actions):
    weight_cf = 1.0 - weight_factual
    total = 0.0
    for xi, ai, ri, ci, rci, cfi in zip(x, a, r, c, rc, cf_action):
        if ci == 1:
            w_f, w_cf = weight_factual, weight_cf
        else:
            w_f, w_cf = 1.0, 0.0
        dm = sum(policy_e(xi, act) * r_hat(xi, act) for act in range(n_actions))
        is_factual = w_f * (policy_e(xi, ai) / policy_b_plus(xi, ai)) * (ri - r_hat(xi, ai))
        if np.isnan(is_factual):
            is_factual = 0.0
        is_cf = 0.0
        if ci == 1 and not np.isnan(rci):
            is_cf = w_cf * (policy_e(xi, cfi) / policy_b_plus(xi, cfi)) * (rci - r_hat(xi, cfi))
        total += dm + is_factual + is_cf
    return total / len(x)


def cdm(x, r_hat_plus, policy_e, n_actions):
    return direct_method(x, r_hat_plus, policy_e, n_actions)


def cdm_is(x, a, r, r_hat_plus, policy_b, policy_e, n_actions):
    return dm_is(x, a, r, r_hat_plus, policy_b, policy_e, n_actions)


def cdm_cis(x, a, r, c, rc, cf_action, r_hat_plus, policy_e, policy_b_plus,
            weight_factual, n_actions):
    return dm_cis(x, a, r, c, rc, cf_action, r_hat_plus, policy_e,
                  policy_b_plus, weight_factual, n_actions)


def dr_combined(x, a, r, c, rc, cf_action, r_hat_plus, policy_b, policy_e,
                n_actions):
    """DR variant that treats each counterfactual annotation as an extra
    sample. For every factual (s, a, r) we add the standard DR term; if a
    counterfactual annotation is available we additionally add a DR term at
    the counterfactual action. The denominator is the total number of samples
    accumulated (factual + counterfactual)."""
    total = 0.0
    n_samples = 0
    for xi, ai, ri, ci, rci, cfi in zip(x, a, r, c, rc, cf_action):
        dm = sum(policy_e(xi, act) * r_hat_plus(xi, act) for act in range(n_actions))
        is_corr = (policy_e(xi, ai) / policy_b(xi, ai)) * (ri - r_hat_plus(xi, ai))
        if np.isnan(is_corr):
            is_corr = 0.0
        total += dm + is_corr
        n_samples += 1
        if ci == 1 and not np.isnan(rci):
            dm_cf = sum(policy_e(xi, act) * r_hat_plus(xi, act) for act in range(n_actions))
            is_cf = (policy_e(xi, cfi) / policy_b(xi, cfi)) * (rci - r_hat_plus(xi, cfi))
            if np.isnan(is_cf):
                is_cf = 0.0
            total += dm_cf + is_cf
            n_samples += 1
    return total / n_samples
