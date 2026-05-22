import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from data_generation import (
    runs, N, d0, policy_b, policy_e, load_data_train_eval, target_policy_value,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import estimators


N_ACTIONS = 2


def _learn_rhat(x_features, r):
    return LinearRegression().fit(x_features, r)


def _learn_rhat_plus(x_features, x_other_features, r, c, rc):
    X_list, y_list = [], []
    for i in range(x_features.shape[0]):
        X_list.append(x_features[i])
        y_list.append(r[i])
        if c[i] == 1:
            X_list.append(x_other_features[i])
            y_list.append(rc[i])
    return LinearRegression().fit(X_list, y_list)


def _corrupt_features(x_features, x_other_features, seed):
    """Misspecification: with 50% probability, replace phi(s, a) with
    phi(s, 1-a) for each train sample, so the reward model learns from
    a corrupted action-feature association."""
    rng = np.random.default_rng(seed=10 + seed)
    out = np.empty_like(x_features)
    for i in range(x_features.shape[0]):
        out[i] = x_features[i] if rng.choice(2) == 0 else x_other_features[i]
    return out


def _compute_policy_b_plus(a, c, weight_factual, policy_b_arr):
    weight_cf = 1.0 - weight_factual
    bar_W = {0: {0: [0.0], 1: [0.0]}, 1: {0: [0.0], 1: [0.0]}}
    for i in range(c.shape[0]):
        if c[i] == 0:
            w_f, w_cf = 1.0, 0.0
        else:
            w_f, w_cf = weight_factual, weight_cf
        ai = int(a[i])
        bar_W[ai][ai].append(w_f)
        bar_W[ai][1 - ai].append(w_cf)
    pb_plus = np.array([
        policy_b_arr[0] * np.nanmean(bar_W[0][0]) + policy_b_arr[1] * np.nanmean(bar_W[1][0]),
        policy_b_arr[0] * np.nanmean(bar_W[0][1]) + policy_b_arr[1] * np.nanmean(bar_W[1][1]),
    ])
    return pb_plus / pb_plus.sum()


def _run_estimators(X, X_other, A, RF, C, RC, X_train, X_train_other,
                    RF_train, C_train, RC_train, pb_arr, pe_arr, ms,
                    weight_factual):
    pb_fn = lambda xi, ai: pb_arr[int(ai)]
    pe_fn = lambda xi, ai: pe_arr[int(ai)]

    names = ["IS", "DM", "DM_IS", "C-IS", "C-DM", "CDM-IS", "DM-CIS",
             "CDM-CIS", "DR_combined"]
    results = {name: [] for name in names}

    for seed in range(runs):
        x_feat = X[seed]
        x_other_feat = X_other[seed]
        a = A[seed].astype(int)
        r = RF[seed]
        c = C[seed].astype(int)
        rc = RC[seed]
        cf_action = (1 - a).astype(int)

        # Pack per-sample features: x_combined[i, action] = phi(s_i, action).
        x_combined = np.zeros((x_feat.shape[0], 2, x_feat.shape[1]))
        for i in range(x_feat.shape[0]):
            x_combined[i, a[i]] = x_feat[i]
            x_combined[i, 1 - a[i]] = x_other_feat[i]

        x_train_feat = X_train[seed]
        x_train_other = X_train_other[seed]
        r_train = RF_train[seed]
        c_train = C_train[seed].astype(int)
        rc_train = RC_train[seed]

        x_train_fit = (_corrupt_features(x_train_feat, x_train_other, seed)
                       if ms else x_train_feat)

        R_hat_model = _learn_rhat(x_train_fit, r_train)
        R_hat_plus_model = _learn_rhat_plus(
            x_train_fit, x_train_other, r_train, c_train, rc_train)

        r_hat = lambda xi, ai, _m=R_hat_model: _m.predict([xi[int(ai)]])[0]
        r_hat_plus = lambda xi, ai, _m=R_hat_plus_model: _m.predict([xi[int(ai)]])[0]

        pbp = _compute_policy_b_plus(a, c, weight_factual, pb_arr)
        pbp_fn = lambda xi, ai, _P=pbp: _P[int(ai)]

        results["IS"].append(estimators.standard_is(x_combined, a, r, pb_fn, pe_fn))
        results["DM"].append(
            estimators.direct_method(x_combined, r_hat, pe_fn, N_ACTIONS))
        results["DM_IS"].append(
            estimators.dm_is(x_combined, a, r, r_hat, pb_fn, pe_fn, N_ACTIONS))
        results["C-IS"].append(
            estimators.cis(x_combined, a, r, c, rc, cf_action, pe_fn, pbp_fn,
                           weight_factual))
        results["C-DM"].append(
            estimators.cdm(x_combined, r_hat_plus, pe_fn, N_ACTIONS))
        results["CDM-IS"].append(
            estimators.cdm_is(x_combined, a, r, r_hat_plus, pb_fn, pe_fn, N_ACTIONS))
        results["DM-CIS"].append(
            estimators.dm_cis(x_combined, a, r, c, rc, cf_action, r_hat, pe_fn,
                              pbp_fn, weight_factual, N_ACTIONS))
        results["CDM-CIS"].append(
            estimators.cdm_cis(x_combined, a, r, c, rc, cf_action, r_hat_plus,
                               pe_fn, pbp_fn, weight_factual, N_ACTIONS))
        results["DR_combined"].append(
            estimators.dr_combined(x_combined, a, r, c, rc, cf_action,
                                   r_hat_plus, pb_fn, pe_fn, N_ACTIONS))
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
    """Evaluate all OPE estimators on HeartSteps under annotation bias/variance.

    Parameters
    ----------
    bias, std : annotation bias and added standard deviation. The CA for the
        non-taken action is drawn ``Normal(true_yc + bias, sigma_R + std)``,
        matching the load_data convention in aistats_heartsteps.ipynb.
    ms : if True, fit the reward model on features whose action coordinate is
        randomly flipped (50%), giving a misspecified reward model.
    weight_counterfactual : weight given to counterfactual annotations in
        the augmented reward / policy. Factual weight is ``1 - weight_counterfactual``.
    missingness : fraction of counterfactual annotations available.
    policy_b_arr, policy_e_arr : optional (2,) behavior/target policies.

    Returns
    -------
    pd.DataFrame with columns ['Approach', 'MSE', 'MSE_lower_bound', 'MSE_higher_bound'].
    """
    pb_arr = policy_b_arr if policy_b_arr is not None else policy_b
    pe_arr = policy_e_arr if policy_e_arr is not None else policy_e
    weight_factual = 1.0 - weight_counterfactual

    (X, X_other, A, RF, C, RC,
     Xt, Xt_other, At, RFt, Ct, RCt) = load_data_train_eval(
        policy_b_arr=pb_arr, bias=bias, std=std, missingness=missingness)
    del At  # not needed; features already encode the action

    results = _run_estimators(
        X, X_other, A, RF, C, RC, Xt, Xt_other, RFt, Ct, RCt,
        pb_arr=pb_arr, pe_arr=pe_arr, ms=ms, weight_factual=weight_factual)

    true_value = target_policy_value(policy_e_arr=pe_arr, n=N)
    rows = []
    for name, values in results.items():
        values_arr = np.asarray(values)
        mse = np.nanmean(np.square(values_arr - true_value))
        mse_low, mse_high = _bootstrap_mse_ci(values_arr, true_value)
        rows.append([name, mse, mse_low, mse_high])
    return pd.DataFrame(rows, columns=['Approach', 'MSE', 'MSE_lower_bound',
                                       'MSE_higher_bound'])
