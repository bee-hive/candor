import os
import sys

import numpy as np
import pandas as pd

from data_generation import (
    runs, d0, policy_b, policy_e, R, ww, load_data_train_eval,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import estimators


N_ACTIONS = 2


def learn_rhat(x, a, r):
    rhat = {0: {0: [], 1: []}, 1: {0: [], 1: []}}
    for i in range(x.shape[0]):
        rhat[int(x[i])][int(a[i])].append(r[i])
    return np.array([[np.mean(rhat[0][0]), np.mean(rhat[0][1])],
                     [np.mean(rhat[1][0]), np.mean(r)]])


def learn_rhat_plus(x, a, r, c, rc):
    rhat_plus = {0: {0: [], 1: []}, 1: {0: [], 1: []}}
    for i in range(x.shape[0]):
        _x, _a = int(x[i]), int(a[i])
        rhat_plus[_x][_a].append(r[i])
        if c[i] == 1:
            rhat_plus[_x][1 - _a].append(rc[i])
    return np.array([[np.mean(rhat_plus[0][0]), np.mean(rhat_plus[0][1])],
                     [np.mean(rhat_plus[1][0]), np.mean(r)]])


def _corrupt_state(x, seed):
    rng = np.random.default_rng(seed=10 + seed)
    out = np.empty(x.shape, dtype=int)
    for i, xi in enumerate(x):
        out[i] = int(xi) if rng.choice(2) == 0 else 1 - int(xi)
    return out


def _compute_policy_b_plus(x, a, c, wf, pb_arr):
    bar_W = {0: {0: {0: [0.0], 1: [0.0]}, 1: {0: [0.0], 1: [0.0]}},
             1: {0: {0: [], 1: []}, 1: {0: [], 1: []}}}
    for i in range(c.shape[0]):
        if c[i] == 0:
            wi, wci = 1.0, 0.0
        else:
            wi, wci = wf, 1.0 - wf
        s, ai = int(x[i]), int(a[i])
        bar_W[s][ai][ai].append(wi)
        bar_W[s][ai][1 - ai].append(wci)

    pbp = np.array([
        [pb_arr[0, 0] * np.nanmean(bar_W[0][0][0]) + pb_arr[0, 1] * np.nanmean(bar_W[0][1][0]),
         pb_arr[0, 0] * np.nanmean(bar_W[0][0][1]) + pb_arr[0, 1] * np.nanmean(bar_W[0][1][1])],
        [pb_arr[1, 0] * ww[1, 0, 0] + pb_arr[1, 1] * ww[1, 1, 0],
         pb_arr[1, 0] * ww[1, 0, 1] + pb_arr[1, 1] * ww[1, 1, 1]],
    ])
    return pbp / pbp.sum(axis=1, keepdims=True)


def _run_estimators(X, A, RF, C, RC, X_train, A_train, RF_train, C_train,
                    RC_train, pb_arr, pe_arr, ms, weight_factual):
    pb_fn = lambda s, ai: pb_arr[int(s), int(ai)]
    pe_fn = lambda s, ai: pe_arr[int(s), int(ai)]

    names = ["IS", "DM", "DM_IS", "C-IS", "C-DM", "CDM-IS", "DM-CIS",
             "CDM-CIS", "DR_combined"]
    results = {name: [] for name in names}
    for seed in range(runs):
        x = X[seed].astype(int)
        a = A[seed].astype(int)
        r = RF[seed]
        c = C[seed].astype(int)
        rc = RC[seed]
        cf_action = (1 - a).astype(int)

        x_fit_src = X_train[seed].astype(int)
        a_fit = A_train[seed].astype(int)
        r_fit = RF_train[seed]
        c_fit = C_train[seed].astype(int)
        rc_fit = RC_train[seed]

        x_fit = _corrupt_state(x_fit_src, seed) if ms else x_fit_src
        R_hat = learn_rhat(x_fit, a_fit, r_fit)
        R_hat_plus = learn_rhat_plus(x_fit, a_fit, r_fit, c_fit, rc_fit)
        pbp = _compute_policy_b_plus(x, a, c, weight_factual, pb_arr)

        r_hat = lambda s, ai, _R=R_hat: _R[int(s), int(ai)]
        r_hat_plus = lambda s, ai, _R=R_hat_plus: _R[int(s), int(ai)]
        pbp_fn = lambda s, ai, _P=pbp: _P[int(s), int(ai)]

        results["IS"].append(estimators.standard_is(x, a, r, pb_fn, pe_fn))
        results["DM"].append(estimators.direct_method(x, r_hat, pe_fn, N_ACTIONS))
        results["DM_IS"].append(
            estimators.dm_is(x, a, r, r_hat, pb_fn, pe_fn, N_ACTIONS))
        results["C-IS"].append(
            estimators.cis(x, a, r, c, rc, cf_action, pe_fn, pbp_fn, weight_factual))
        results["C-DM"].append(estimators.cdm(x, r_hat_plus, pe_fn, N_ACTIONS))
        results["CDM-IS"].append(
            estimators.cdm_is(x, a, r, r_hat_plus, pb_fn, pe_fn, N_ACTIONS))
        results["DM-CIS"].append(
            estimators.dm_cis(x, a, r, c, rc, cf_action, r_hat, pe_fn,
                              pbp_fn, weight_factual, N_ACTIONS))
        results["CDM-CIS"].append(
            estimators.cdm_cis(x, a, r, c, rc, cf_action, r_hat_plus, pe_fn,
                               pbp_fn, weight_factual, N_ACTIONS))
        results["DR_combined"].append(
            estimators.dr_combined(x, a, r, c, rc, cf_action, r_hat_plus,
                                   pb_fn, pe_fn, N_ACTIONS))
    return results


def _bootstrap_mse_ci(values, true_value, alpha=0.05,
                      n_bootstrap=1000, n_resamples=100):
    se = np.square(np.asarray(values) - true_value)
    samples = np.random.choice(se, (n_resamples, n_bootstrap), replace=True)
    mse_samples = np.nanmean(samples, axis=1)
    lo = np.percentile(mse_samples, 100 * alpha / 2)
    hi = np.percentile(mse_samples, 100 * (1 - alpha / 2))
    return lo, hi


def run_experiment(bias=0.0, std=0.0, ms=False, weight_counterfactual=0.3,
                   missingness=1.0, policy_b_arr=None, policy_e_arr=None):
    """Evaluate all OPE estimators under annotation bias/variance.

    Parameters
    ----------
    bias, std : annotation bias and standard deviation.
    ms : if True, reward models are fit on 50/50-flipped state labels
        (misspecified reward model).
    weight_counterfactual : weight given to counterfactual annotations in
        the augmented reward / policy. Factual weight is ``1 - weight_counterfactual``.
    missingness : fraction of counterfactual annotations available.
    policy_b_arr, policy_e_arr : optional (2, 2) behavior/target policies.

    Returns
    -------
    pd.DataFrame with columns ['Approach', 'MSE', 'MSE_lower_bound', 'MSE_higher_bound'].
    """
    pb_arr = policy_b_arr if policy_b_arr is not None else policy_b
    pe_arr = policy_e_arr if policy_e_arr is not None else policy_e
    weight_factual = 1.0 - weight_counterfactual

    X, A, RF, C, RC, Xt, At, Rt, Ct, RCt = load_data_train_eval(
        policy_b_arr=pb_arr, bias=bias, std=std, missingness=missingness)
    results = _run_estimators(
        X, A, RF, C, RC, Xt, At, Rt, Ct, RCt,
        pb_arr=pb_arr, pe_arr=pe_arr, ms=ms, weight_factual=weight_factual)

    true_value = d0 @ np.sum(pe_arr * R, axis=1)
    rows = []
    for name, values in results.items():
        values_arr = np.asarray(values)
        mse = np.nanmean(np.square(values_arr - true_value))
        mse_low, mse_high = _bootstrap_mse_ci(values_arr, true_value)
        rows.append([name, mse, mse_low, mse_high])
    return pd.DataFrame(rows, columns=['Approach', 'MSE', 'MSE_lower_bound', 'MSE_higher_bound'])
