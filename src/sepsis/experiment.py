"""Single sepsis experiment driver, analogous to toy_bandit/experiment.py.

``run_experiment(bias, std, ms, ...)`` evaluates eight OPE estimators
against pre-generated data and returns MSE + bootstrap CIs. All estimators
come from ``src/estimators.py``; the sepsis-specific work is wrapping the
reward models and policies as callables. The shared ``dr_combined`` is
intentionally not run for sepsis (see ``_run_one_seed`` for the reason).

The data files are produced once via ``data_generation.generate_all``
(see that module's docstring) and read by ``data_generation.load_data``.
"""
import os
import sys

import numpy as np
import pandas as pd

from data_generation import (
    nA, runs, policy_e as DEFAULT_POLICY_E, behavior_policies,
    load_data, load_true_values,
    vec_to_state, convert_to_ws,
    get_state_action_feature, truncate_prediction,
    learn_rhat, learn_rhat_plus, learn_rhat_ws, learn_rhat_plus_ws,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import estimators


# d = 21 (state-vector dimension) — re-imported lazily to avoid circular issues
from data_generation import d as STATE_DIM


def _make_r_hat(R_hat_model, ms):
    """Wrap a fitted LinearRegression as a callable ``(state_vec, action) -> float``.

    ``ms=True``: high-dim one-hot state-action featurization (misspecified).
    ``ms=False``: 2-d ``convert_to_ws`` featurization (well-specified, slow:
    each call performs an MDP transition under the hood).
    """
    if ms:
        def r_hat(xi, ai, _m=R_hat_model):
            sa = np.expand_dims(get_state_action_feature(xi, int(ai)), axis=0)
            return float(truncate_prediction(_m.predict(sa).item()))
        return r_hat

    def r_hat(xi, ai, _m=R_hat_model):
        return float(_m.predict(convert_to_ws(vec_to_state(xi), int(ai))).item())
    return r_hat


def _policy_b_plus(weight_factual, policy_b_arr):
    """Augmented behavior policy used by the C-IS family. Each ``act`` gets
    the factual probability mass plus an equal share of the counterfactual
    weight from every other action."""
    bar_w_cf = (1 - weight_factual) / (nA - 1)
    pbp = np.zeros_like(policy_b_arr)
    for act in range(nA):
        cf_mass = bar_w_cf * sum(policy_b_arr[o] for o in range(nA) if o != act)
        pbp[act] = weight_factual * policy_b_arr[act] + cf_mass
    return pbp / pbp.sum()


def _run_one_seed(seed, bias, std, ms, weight_factual, missingness,
                  behaviorPol_idx, policy_e_arr, data_dir):
    XA, XA_act, _, R, CA, _, CA_act, G = load_data(
        seed, bias=bias, std=std, behaviorPol_idx=behaviorPol_idx,
        data_dir=data_dir, missingness=missingness)
    XA_act = XA_act.astype(int)
    CA_act = CA_act.astype(int)
    n_samples = XA.shape[0]
    policy_b_arr = behavior_policies[behaviorPol_idx]

    # Per-sample state vector: pull the state out of XA's action slot.
    x = np.zeros((n_samples, STATE_DIM))
    for i in range(n_samples):
        x[i] = XA[i].reshape((nA, -1))[XA_act[i]]

    c = (~np.isnan(G)).astype(int)
    rc = G  # NaN where c == 0

    if ms:
        R_hat_model = learn_rhat(XA, R)
        R_hat_plus_model = learn_rhat_plus(XA, R, CA, G)
    else:
        R_hat_model = learn_rhat_ws(XA, R)
        R_hat_plus_model = learn_rhat_plus_ws(XA, R, CA, G)

    r_hat = _make_r_hat(R_hat_model, ms)
    r_hat_plus = _make_r_hat(R_hat_plus_model, ms)

    pb_fn = lambda _xi, ai: policy_b_arr[int(ai)]
    pe_fn = lambda _xi, ai: policy_e_arr[int(ai)]
    pbp = _policy_b_plus(weight_factual, policy_b_arr)
    pbp_fn = lambda _xi, ai, _P=pbp: _P[int(ai)]

    # Note: ``DR_combined`` is intentionally omitted. The sepsis behavior
    # policies place probability 0 on some actions (e.g., policy_b1[5..7]=0),
    # which makes the standard CF IS correction divide-by-zero whenever the
    # counterfactual action falls on a zero-support action under pi_b. The
    # C-IS family handles this via the augmented ``policy_b_plus``.
    return {
        "IS": estimators.standard_is(x, XA_act, R, pb_fn, pe_fn),
        "DM": estimators.direct_method(x, r_hat, pe_fn, nA),
        "DM_IS": estimators.dm_is(x, XA_act, R, r_hat, pb_fn, pe_fn, nA),
        "C-IS": estimators.cis(x, XA_act, R, c, rc, CA_act, pe_fn, pbp_fn,
                               weight_factual),
        "C-DM": estimators.cdm(x, r_hat_plus, pe_fn, nA),
        "CDM-IS": estimators.cdm_is(x, XA_act, R, r_hat_plus, pb_fn, pe_fn, nA),
        "DM-CIS": estimators.dm_cis(x, XA_act, R, c, rc, CA_act, r_hat, pe_fn,
                                    pbp_fn, weight_factual, nA),
        "CDM-CIS": estimators.cdm_cis(x, XA_act, R, c, rc, CA_act, r_hat_plus,
                                      pe_fn, pbp_fn, weight_factual, nA),
    }


def _bootstrap_mse_ci(values, true_values, alpha=0.05,
                      n_bootstrap=1000, n_resamples=100):
    se = np.square(np.asarray(values) - np.asarray(true_values))
    samples = np.random.choice(se, (n_resamples, n_bootstrap), replace=True)
    mse_samples = np.nanmean(samples, axis=1)
    lo = np.percentile(mse_samples, 100 * alpha / 2)
    hi = np.percentile(mse_samples, 100 * (1 - alpha / 2))
    return lo, hi


def run_experiment(bias=0.0, std=0.0, ms=False, weight_counterfactual=0.5,
                   missingness=1.0, behaviorPol_idx=0, data_dir=None,
                   policy_e_arr=None):
    """Evaluate the eight OPE estimators on the sepsis bandit data.

    Parameters
    ----------
    bias, std : annotation bias and added std encoded in the file suffix
        ``Ca_bias={bias}_variance={std}``.
    ms : if True, fit the reward model on the high-dim state-action features
        (misspecified). If False, fit on the 2-d ``convert_to_ws`` features
        (well-specified).
    weight_counterfactual : weight for counterfactual annotations. Factual
        weight is ``1 - weight_counterfactual``.
    missingness : fraction of counterfactual annotations to keep (applied
        post-hoc by ``load_data``).
    behaviorPol_idx : index into ``behavior_policies``.
    data_dir : override ``data_generation.DATA_DIR``.
    policy_e_arr : optional (nA,) target policy. Defaults to module-level.

    Returns
    -------
    DataFrame with columns ['Approach', 'MSE', 'MSE_lower_bound',
    'MSE_higher_bound'].
    """
    weight_factual = 1.0 - weight_counterfactual
    pe_arr = policy_e_arr if policy_e_arr is not None else DEFAULT_POLICY_E
    true_values = load_true_values(data_dir=data_dir)

    names = ["IS", "DM", "DM_IS", "C-IS", "C-DM", "CDM-IS", "DM-CIS",
             "CDM-CIS"]
    results = {name: [] for name in names}
    for seed in range(runs):
        per_seed = _run_one_seed(seed, bias, std, ms, weight_factual,
                                 missingness, behaviorPol_idx, pe_arr,
                                 data_dir)
        for name in names:
            results[name].append(per_seed[name])

    rows = []
    for name in names:
        vals = np.asarray(results[name])
        mse = np.nanmean(np.square(vals - true_values))
        lo, hi = _bootstrap_mse_ci(vals, true_values)
        rows.append([name, mse, lo, hi])
    return pd.DataFrame(rows, columns=['Approach', 'MSE', 'MSE_lower_bound',
                                       'MSE_higher_bound'])
