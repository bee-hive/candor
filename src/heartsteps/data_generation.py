import numpy as np


class HeartStepsEnvironment:
    """HeartSteps simulator. `Theta_offset` parameterizes a biased reward variant
    used in the misspecified-reward experiments; pass biased=True to env.pull
    to return rewards under the biased Theta."""

    def __init__(self, T, user_type, y0, Theta_offset=0):
        self.user_type = user_type
        self.T = T
        self.y0 = y0
        self.Theta = [-0.04, 0.9999, 0.3]
        self.Theta_biased = [-0.04, 0.9999, 0.3 - Theta_offset]

        if user_type == "active":
            self.group = 0
        elif user_type == "inactive":
            self.group = 1
        else:
            raise ValueError("user_type must be 'active' or 'inactive'")
        self.num_notifications = 0
        self.previous_steps = self.y0

    def pull(self, a, n_notif, prev_y, T, noise=0, biased=False):
        p_sa = self.phi_sa(a=a, group=self.group, n_notif=n_notif + a, prev_y=prev_y, T=T)
        if biased:
            y_it = np.dot(p_sa, self.Theta_biased) + noise
        else:
            y_it = np.dot(p_sa, self.Theta) + noise
        return y_it, p_sa

    def reset(self):
        self.num_notifications = 0
        self.previous_steps = self.y0

    def phi_sa(self, a, group, n_notif, prev_y, T):
        inv_sigmoid = lambda x: np.exp(-x) / (1 + np.exp(-x))
        map_t = lambda x: (x - T / 0.95) * (5 / T)
        s = inv_sigmoid(map_t(n_notif))
        group_te = [0]
        group_te[group] = s * a
        phi = [1, prev_y] + group_te
        return np.asarray(phi)


runs = 50
N = 200
d0 = np.arange(10, 90)
policy_b = np.array([0.9, 0.1])
policy_e = np.array([0.1, 0.9])
sigma_R = 0.5


def load_data_train_eval(policy_b_arr=None, bias=0.0, std=0.0,
                          missingness=1.0, n=None):
    """Generate a paired (eval, train) HeartSteps dataset for the
    bias/variance/missingness sweep. Each sample is independent: the env is
    reset per sample, so only the sampled initial state ``y0`` varies and the
    setting reduces to a contextual bandit.

    Parameters
    ----------
    policy_b_arr : (2,) array. Defaults to module-level ``policy_b``. The
        target policy doesn't appear here — it's only used at evaluation time.
    bias, std : floats. Counterfactual-annotation bias and added std. The CA
        for action ``1 - a`` is drawn ``Normal(true_yc + bias, sigma_R + std)``
        (matching the load_data convention in aistats_heartsteps.ipynb).
    missingness : float in [0, 1]. Probability of a counterfactual annotation
        being available (1.0 = every sample has a CA; 0.0 = none).
    n : int. Number of samples per seed; defaults to module-level ``N``.

    Returns
    -------
    X, X_other, A, RF, C, RC, X_train, X_train_other, A_train, RF_train,
    C_train, RC_train : np.ndarray
        ``X`` and ``X_other`` are phi(s, a) and phi(s, 1-a) for each eval
        sample (shape ``(runs, n, 3)``). The ``_train`` arrays are the same
        for the train split. ``A``/``A_train`` have shape ``(runs, n)``;
        ``RF``/``RF_train``/``C``/``C_train``/``RC``/``RC_train`` likewise.
    """
    pb = policy_b_arr if policy_b_arr is not None else policy_b
    pc = missingness
    n_samples = n if n is not None else N

    X, X_other, A, RF, C, RC = [], [], [], [], [], []
    Xt, Xt_other, At, RFt, Ct, RCt = [], [], [], [], [], []

    for seed in range(runs):
        rng = np.random.default_rng(seed=10 + seed)
        rng_c = np.random.default_rng(seed=100000 + seed)

        def _draw_split():
            x_, xo_, a_, r_, c_, rc_ = [], [], [], [], [], []
            for _ in range(n_samples):
                y0 = rng.choice(d0)
                env = HeartStepsEnvironment(T=n_samples, user_type='active', y0=y0)
                prev_y = env.y0
                a = int(rng.choice(2, p=pb))
                c = int(rng_c.choice(2, p=[1 - pc, pc]))
                _y, _psa = env.pull(a, prev_y=prev_y, n_notif=env.num_notifications, T=n_samples)
                _yo, _psa_o = env.pull(1 - a, prev_y=prev_y, n_notif=env.num_notifications, T=n_samples)
                x_.append(_psa)
                xo_.append(_psa_o)
                a_.append(a)
                r_.append(rng.normal(_y, sigma_R))
                c_.append(c)
                if c == 1:
                    rc_.append(rng_c.normal(_yo + bias, sigma_R + std))
                else:
                    rc_.append(np.nan)
            return x_, xo_, a_, r_, c_, rc_

        xe, xoe, ae, re, ce, rce = _draw_split()
        X.append(xe); X_other.append(xoe); A.append(ae)
        RF.append(re); C.append(ce); RC.append(rce)

        xt, xot, at, rt, ct, rct = _draw_split()
        Xt.append(xt); Xt_other.append(xot); At.append(at)
        RFt.append(rt); Ct.append(ct); RCt.append(rct)

    return (np.asarray(X), np.asarray(X_other), np.asarray(A),
            np.asarray(RF), np.asarray(C), np.asarray(RC),
            np.asarray(Xt), np.asarray(Xt_other), np.asarray(At),
            np.asarray(RFt), np.asarray(Ct), np.asarray(RCt))


def target_policy_value(policy_e_arr=None, n=None):
    """Analytic E_{y0 ~ Uniform(d0)} [ E_{a ~ pe} [ R(y0, a) ] ] using the
    noise-free env. The truth used for MSE in run_experiment."""
    pe = policy_e_arr if policy_e_arr is not None else policy_e
    n_samples = n if n is not None else N
    v = 0.0
    for s_init in d0:
        env = HeartStepsEnvironment(T=n_samples, user_type='active', y0=s_init)
        prev_y = env.y0
        _y0, _ = env.pull(0, prev_y=prev_y, n_notif=env.num_notifications, T=n_samples)
        _y1, _ = env.pull(1, prev_y=prev_y, n_notif=env.num_notifications, T=n_samples)
        v += pe[0] * _y0 + pe[1] * _y1
    return v / len(d0)
