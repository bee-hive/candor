import numpy as np

runs = 100
N = 400
d0 = np.array([0.5, 0.5])
policy_b = np.array([[0.5, 0.5], [1., 0.]])
policy_e = np.array([[0.1, 0.9], [1., 0.]])
R = np.array([[1., 2.], [0., 0.]])
sigma = np.array([[0.5, 0.5], [0.5, 0.5]])
ww = np.array([
    [[0.5, 0.5], [0.5, 0.5]],
    [[1, 0], [0, 1]],
])


def load_data_train_eval(policy_b_arr=None, bias=0.0, std=0.0,
                          missingness=1.0, n=None):
    """Generate a paired (eval, train) dataset for the misspecified-reward
    experiments. The eval set is what the OPE estimators are computed on; the
    train set is what the reward model is fit on. Both come from the same
    distribution but use independent random draws.

    Parameters
    ----------
    policy_b_arr : (2, 2) array. Defaults to module-level ``policy_b``. The
        target policy (``policy_e``) doesn't appear here — it's only used at
        evaluation time, not for generating data.
    bias, std : floats. Counterfactual-annotation bias and added std.
    missingness : float in [0, 1]. Pc value for state 0 (state 1 never has
        counterfactuals). 1.0 = every sample has a CA; 0.0 = no CAs.
    n : int. Number of samples per seed; defaults to module-level ``N``.

    Returns
    -------
    X, A, RF, C, RC, X_train, A_train, RF_train, C_train, RC_train : np.ndarray
        Each of shape ``(runs, n)``.
    """
    pb = policy_b_arr if policy_b_arr is not None else policy_b
    pc_local = np.array([[missingness, missingness], [0.0, 0.0]])
    n_samples = n if n is not None else N

    X, A, RF_, C_, RC_ = [], [], [], [], []
    Xt, At, Rt, Ct, RCt = [], [], [], [], []
    for seed in range(runs):
        rng = np.random.default_rng(seed=10 + seed)
        rng_c = np.random.default_rng(seed=100000 + seed)

        x = rng.choice(len(d0), size=n_samples, p=d0)
        a = np.array([rng.choice(2, p=pb[xi]) for xi in x])
        r = np.array([rng.normal(R[xi, ai], sigma[xi, ai]) for xi, ai in zip(x, a)])
        c = np.array([rng_c.choice(2, p=[1 - pc_local[xi, ai], pc_local[xi, ai]]) for xi, ai in zip(x, a)])
        rc = np.array([rng_c.normal(R[xi, 1 - ai] + bias, sigma[xi, 1 - ai] + std) for xi, ai in zip(x, a)])
        rc[c == 0] = np.nan
        X.append(x); A.append(a); RF_.append(r); C_.append(c); RC_.append(rc)

        x_train = rng.choice(len(d0), size=n_samples, p=d0)
        a_train = np.array([rng.choice(2, p=pb[xi]) for xi in x_train])
        r_train = np.array([rng.normal(R[xi, ai], sigma[xi, ai]) for xi, ai in zip(x_train, a_train)])
        c_train = np.array([rng_c.choice(2, p=[1 - pc_local[xi, ai], pc_local[xi, ai]]) for xi, ai in zip(x_train, a_train)])
        rc_train = np.array([rng_c.normal(R[xi, 1 - ai] + bias, sigma[xi, 1 - ai] + std) for xi, ai in zip(x_train, a_train)])
        rc_train[c_train == 0] = np.nan
        Xt.append(x_train); At.append(a_train); Rt.append(r_train); Ct.append(c_train); RCt.append(rc_train)

    return (np.asarray(X), np.asarray(A), np.asarray(RF_), np.asarray(C_), np.asarray(RC_),
            np.asarray(Xt), np.asarray(At), np.asarray(Rt), np.asarray(Ct), np.asarray(RCt))
