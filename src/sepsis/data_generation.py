"""Sepsis contextual-bandit dataset: generation + loading.

This module is self-contained. The pre-generation pipeline (`generate_all`)
uses the sepsisSimDiabetes MDP under ``data_generation/`` to write per-seed
``.npy`` files; the loading pipeline (`load_data`) reads them back during the
OPE experiments.

Typical usage
-------------
First, generate the data once (this takes a while):

    python -c "from data_generation import generate_all; \
               generate_all(output_dir='datagen/cb')"

Then run an experiment:

    from experiment import run_experiment
    df = run_experiment(bias=0.1, std=0.1, behaviorPol_idx=0,
                        data_dir='datagen/cb')

Files written by ``generate_all`` (for each ``pol_idx`` in 0..5 and
``seed`` in 0..runs-1):

    <output_dir>/<pol_idx>/<seed>-XA.npy            # state-action features
    <output_dir>/<pol_idx>/<seed>-A.npy             # factual actions
    <output_dir>/<pol_idx>/<seed>-R.npy             # factual rewards
    <output_dir>/<pol_idx>/<seed>-NX.npy            # next-state 2-d (num_abnormal, on_treatment)
    <output_dir>/<pol_idx>/<seed>-X_IDX.npy         # state indices (for CA generation)
    <output_dir>/<pol_idx>/<seed>-Ca_bias={b}_variance={v}.npy
    <output_dir>/<pol_idx>/<seed>-Ca_act_bias={b}_variance={v}.npy
    <output_dir>/<pol_idx>/<seed>-NxCa_bias={b}_variance={v}.npy
    <output_dir>/<pol_idx>/<seed>-G_bias={b}_variance={v}.npy
    <output_dir>/true_values.npy                    # per-seed true V^pi_e
"""
import os
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data_generation'))
from sepsisSimDiabetes.State import State
from sepsisSimDiabetes.Action import Action
import sepsisSimDiabetes.MDP as simulator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

nS, nA = 1442, 8
d = 21
runs = 50
NSIMSAMPS = 700  # samples per seed (matches the data-prep notebook)
FACTUAL_NOISE_SIGMA = 0.05  # noise added to factual rewards
CA_BASE_NOISE_SIGMA = 0.005  # base noise on counterfactual annotations

policy_e = np.asarray([0.3, 0.2, 0.0, 0.0, 0.2, 0.1, 0.1, 0.1])
behavior_policies = [
    np.asarray([0.1, 0.1, 0.4, 0.3, 0.1, 0.0, 0.0, 0.0]),
    np.asarray([0.1, 0.1, 0.4, 0.2, 0.1, 0.1, 0.0, 0.0]),
    np.asarray([0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.0, 0.1]),
    np.asarray([0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.asarray([0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]),
    np.asarray([0.3, 0.1, 0.2, 0.0, 0.1, 0.1, 0.1, 0.1]),
]

DEFAULT_BIAS_VALUES = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_VARIANCE_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Override per-call via the data_dir kwarg or by setting the env var.
DATA_DIR = os.environ.get("SEPSIS_DATA_DIR", "./datagen")


# ---------------------------------------------------------------------------
# Featurization / state helpers
# ---------------------------------------------------------------------------

def get_state_action_feature(x_s, a):
    """One-hot state across action slots: (nA * d,) with x_s placed in slot a."""
    x_sa = np.zeros((nA, d))
    x_sa[a, :] = x_s
    return x_sa.flatten()


def state_to_vec(s):
    if s.diabetic_idx == 2:
        s.diabetic_idx = 1  # death state collapses onto diabetic=1
    vec = []
    for size, val in [(2, s.diabetic_idx), (3, s.hr_state), (3, s.sysbp_state),
                      (2, s.percoxyg_state), (5, s.glucose_state),
                      (2, s.antibiotic_state), (2, s.vaso_state),
                      (2, s.vent_state)]:
        vec.extend(np.eye(size)[val].tolist())
    return vec


def vec_to_state(_s):
    diabetic_idx = np.nonzero(_s[:2])[0].item()
    hr = np.nonzero(_s[2:5])[0].item()
    sysbp = np.nonzero(_s[5:8])[0].item()
    percoxy = np.nonzero(_s[8:10])[0].item()
    glucose = np.nonzero(_s[10:15])[0].item()
    antibiotics = np.nonzero(_s[15:17])[0].item()
    vaso = np.nonzero(_s[17:19])[0].item()
    vent = np.nonzero(_s[19:])[0].item()
    state_categs = np.array([hr, sysbp, percoxy, glucose, antibiotics, vaso, vent])
    return State(diabetic_idx=diabetic_idx, state_categs=state_categs)


def convert_to_ws(s, a):
    """Project (state, action) to the 2-d well-specified feature
    ``(num_abnormal, on_treatment)`` after applying ``MDP.transition(a)``."""
    _ai = Action(action_idx=a)
    mdp = simulator.MDP()
    _ = mdp.get_new_state(state_idx=s.get_state_idx(), idx_type='full')
    _ = mdp.transition(_ai)
    s_prime = mdp.state
    return np.expand_dims(
        [s_prime.get_num_abnormal(), int(s_prime.on_treatment())], axis=0)


def truncate_prediction(pred):
    if pred < 0:
        return max(-10, pred)
    if pred > 0:
        return min(10, pred)
    return pred


def get_annotation(state_idx, ca):
    """Run the MDP from ``state_idx`` under counterfactual action ``ca`` and
    return ``(nx_2d, reward)`` where ``nx_2d`` is the 2-d
    ``(num_abnormal, on_treatment)`` representation of the next state."""
    _si = State(state_idx=state_idx, idx_type='full')
    _ai = Action(action_idx=ca)
    mdp = simulator.MDP()
    _ = mdp.get_new_state(idx_type='full', state_idx=_si.get_state_idx())
    g_i = mdp.transition(_ai)
    nx_i = [mdp.state.get_num_abnormal(), int(mdp.state.on_treatment())]
    return nx_i, g_i


# ---------------------------------------------------------------------------
# Generation pipeline (port of src/sepsisSim/data-prep/sepsisSim_contextual_bandit.ipynb)
# ---------------------------------------------------------------------------

def _ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def generate_factual(behaviorPol_idx, output_dir, n_runs=runs,
                     nsimsamps=NSIMSAMPS, show_progress=True):
    """Generate factual ``(XA, A, R, NX, X_IDX)`` for one behavior policy.

    Writes ``<output_dir>/<behaviorPol_idx>/<seed>-{XA,A,R,NX,X_IDX}.npy``
    for ``seed in range(n_runs)``. Each file has ``nsimsamps`` rows.
    """
    pol = behavior_policies[behaviorPol_idx]
    pol_dir = os.path.join(output_dir, str(behaviorPol_idx))
    _ensure_dir(pol_dir)

    iterator = range(n_runs)
    if show_progress:
        iterator = tqdm(iterator, desc=f"factual pol={behaviorPol_idx}")
    for seed in iterator:
        rng = np.random.default_rng(seed=10 + seed)
        XA = np.zeros((nsimsamps, nA * d))
        A = np.zeros(nsimsamps, dtype=int)
        R = np.zeros(nsimsamps)
        NX = np.zeros((nsimsamps, 2))
        X_IDX = np.zeros(nsimsamps, dtype=int)
        for i in range(nsimsamps):
            mdp = simulator.MDP()
            state_idx = int(rng.integers(nS))
            s = mdp.get_new_state(state_idx=state_idx, idx_type='full')
            action_idx = int(rng.choice(nA, p=pol))
            a = Action(action_idx=action_idx)
            reward = mdp.transition(a)
            XA[i] = get_state_action_feature(state_to_vec(s), action_idx)
            A[i] = action_idx
            R[i] = rng.normal(reward, FACTUAL_NOISE_SIGMA)
            NX[i] = [mdp.state.get_num_abnormal(), int(mdp.state.on_treatment())]
            X_IDX[i] = s.get_state_idx()

        np.save(os.path.join(pol_dir, f"{seed}-XA.npy"), XA)
        np.save(os.path.join(pol_dir, f"{seed}-A.npy"), A)
        np.save(os.path.join(pol_dir, f"{seed}-R.npy"), R)
        np.save(os.path.join(pol_dir, f"{seed}-NX.npy"), NX)
        np.save(os.path.join(pol_dir, f"{seed}-X_IDX.npy"), X_IDX)


def generate_kitchen_sink_ca(behaviorPol_idx, output_dir,
                             bias_values=None, variance_values=None,
                             n_runs=runs, show_progress=True):
    """Generate counterfactual annotations for all (bias, variance) combos.

    Requires ``generate_factual(behaviorPol_idx, output_dir, n_runs)`` to
    have already been run.

    Writes for each ``(bias, variance, seed)``:
        <output_dir>/<behaviorPol_idx>/<seed>-Ca_bias={b}_variance={v}.npy
        <output_dir>/<behaviorPol_idx>/<seed>-Ca_act_bias={b}_variance={v}.npy
        <output_dir>/<behaviorPol_idx>/<seed>-NxCa_bias={b}_variance={v}.npy
        <output_dir>/<behaviorPol_idx>/<seed>-G_bias={b}_variance={v}.npy
    """
    bias_values = bias_values if bias_values is not None else DEFAULT_BIAS_VALUES
    variance_values = variance_values if variance_values is not None else DEFAULT_VARIANCE_VALUES
    pol_dir = os.path.join(output_dir, str(behaviorPol_idx))
    _ensure_dir(pol_dir)

    outer = bias_values
    if show_progress:
        outer = tqdm(bias_values, desc=f"kitchen-sink pol={behaviorPol_idx}")
    for b in outer:
        for v in variance_values:
            for seed in range(n_runs):
                rng = np.random.default_rng(seed=100000 + seed + int(round(b * 1000)) * 13 + int(round(v * 1000)))
                XA = np.load(os.path.join(pol_dir, f"{seed}-XA.npy"))
                X_IDX = np.load(os.path.join(pol_dir, f"{seed}-X_IDX.npy"))
                A = np.load(os.path.join(pol_dir, f"{seed}-A.npy"))
                n = XA.shape[0]
                CA = np.zeros_like(XA)
                CA_act = np.zeros(n, dtype=int)
                NX_CA = np.zeros((n, 2))
                G = np.zeros(n)
                for i in range(n):
                    _a = int(A[i])
                    _x = XA[i].reshape((nA, -1))[_a]  # state vector without action slot
                    _x_idx = int(X_IDX[i])
                    possible_actions = [act for act in range(nA) if act != _a]
                    ca = int(rng.choice(possible_actions))
                    nx_i, g_i = get_annotation(_x_idx, ca)
                    CA[i, :] = get_state_action_feature(_x, ca)
                    CA_act[i] = ca
                    NX_CA[i, :] = nx_i
                    G[i] = rng.normal(g_i + b, CA_BASE_NOISE_SIGMA + v)

                suffix = f"bias={b}_variance={v}"
                np.save(os.path.join(pol_dir, f"{seed}-Ca_act_{suffix}.npy"), CA_act)
                np.save(os.path.join(pol_dir, f"{seed}-Ca_{suffix}.npy"), CA)
                np.save(os.path.join(pol_dir, f"{seed}-G_{suffix}.npy"), G)
                np.save(os.path.join(pol_dir, f"{seed}-NxCa_{suffix}.npy"), NX_CA)


def compute_true_values(output_dir, n_runs=runs, show_progress=True):
    """Estimate V^{pi_e} for each seed via Monte Carlo over all states (the
    noise comes from sampling the factual reward), then save to
    ``<output_dir>/true_values.npy``."""
    _ensure_dir(output_dir)
    true_values = np.zeros(n_runs)
    iterator = range(n_runs)
    if show_progress:
        iterator = tqdm(iterator, desc="true_values")
    for seed in iterator:
        rng = np.random.default_rng(seed=200000 + seed)
        state_values = np.zeros(nS)
        for i in range(nS):
            val_state = 0.0
            for a in range(nA):
                mdp = simulator.MDP()
                _ = mdp.get_new_state(state_idx=i, idx_type='full')
                r = mdp.transition(Action(action_idx=a))
                val_state += policy_e[a] * rng.normal(r, FACTUAL_NOISE_SIGMA)
            state_values[i] = val_state
        true_values[seed] = np.mean(state_values)
    np.save(os.path.join(output_dir, "true_values.npy"), true_values)
    return true_values


def generate_all(output_dir, behaviorPol_indices=None, bias_values=None,
                 variance_values=None, n_runs=runs, nsimsamps=NSIMSAMPS,
                 show_progress=True):
    """Run the full generation pipeline: factual data for each behavior
    policy, all kitchen-sink counterfactual annotations, and true values."""
    if behaviorPol_indices is None:
        behaviorPol_indices = list(range(len(behavior_policies)))

    for pol_idx in behaviorPol_indices:
        generate_factual(pol_idx, output_dir, n_runs=n_runs,
                         nsimsamps=nsimsamps, show_progress=show_progress)
        generate_kitchen_sink_ca(pol_idx, output_dir, bias_values=bias_values,
                                 variance_values=variance_values,
                                 n_runs=n_runs, show_progress=show_progress)
    compute_true_values(output_dir, n_runs=n_runs, show_progress=show_progress)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _path(data_dir, behaviorPol_idx, seed, name):
    return os.path.join(data_dir, str(behaviorPol_idx), f"{seed}-{name}.npy")


def load_data(seed, bias, std, behaviorPol_idx, data_dir=None,
              missingness=1.0):
    """Load one seed's (factual + counterfactual) sample for a given
    annotation ``(bias, std)`` and behavior-policy index.

    If ``missingness < 1.0``, that fraction of CA samples is zeroed out
    (``CA`` row set to zero, ``G`` set to ``nan``).
    """
    dd = data_dir if data_dir is not None else DATA_DIR
    pol_dir = os.path.join(dd, str(behaviorPol_idx))

    XA = np.load(_path(dd, behaviorPol_idx, seed, 'XA'))
    NX = np.load(_path(dd, behaviorPol_idx, seed, 'NX'))
    XA_act = np.load(_path(dd, behaviorPol_idx, seed, 'A'))
    R = np.load(_path(dd, behaviorPol_idx, seed, 'R'))

    suffix = f"bias={bias}_variance={std}"
    CA = np.load(os.path.join(pol_dir, f"{seed}-Ca_{suffix}.npy"))
    NX_CA = np.load(os.path.join(pol_dir, f"{seed}-NxCa_{suffix}.npy"))
    CA_act = np.load(os.path.join(pol_dir, f"{seed}-Ca_act_{suffix}.npy"))
    G = np.load(os.path.join(pol_dir, f"{seed}-G_{suffix}.npy"))

    if missingness < 1.0:
        rng = np.random.default_rng(seed=300000 + seed)
        drop = rng.random(CA.shape[0]) > missingness
        CA = CA.copy()
        G = G.copy().astype(float)
        CA[drop] = 0
        G[drop] = np.nan

    return XA, XA_act, NX, R, CA, NX_CA, CA_act, G


def load_true_values(data_dir=None):
    """Load the (runs,) array of per-seed true V^{pi_e} values."""
    dd = data_dir if data_dir is not None else DATA_DIR
    return np.load(os.path.join(dd, 'true_values.npy'))


# ---------------------------------------------------------------------------
# Reward models
# ---------------------------------------------------------------------------

def learn_rhat(XA, R):
    return LinearRegression().fit(XA, R)


def learn_rhat_plus(XA, R, CA, G):
    mask = ~np.isnan(G)
    X = np.vstack((XA, CA[mask]))
    Y = np.hstack((R, G[mask]))
    return LinearRegression().fit(X, Y)


def learn_rhat_ws(XA, R):
    XA_ws = np.zeros((XA.shape[0], 2))
    for i in range(XA.shape[0]):
        _a = np.nonzero(XA[i].reshape((nA, -1)))[0][0]
        _s = XA[i].reshape((nA, -1))[_a]
        XA_ws[i, :] = convert_to_ws(vec_to_state(_s), _a)[0]
    return LinearRegression().fit(XA_ws, R)


def learn_rhat_plus_ws(XA, R, CA, G):
    mask = ~np.isnan(G)
    XA_ws = np.zeros((XA.shape[0], 2))
    for i in range(XA.shape[0]):
        _a = np.nonzero(XA[i].reshape((nA, -1)))[0][0]
        _s = XA[i].reshape((nA, -1))[_a]
        XA_ws[i, :] = convert_to_ws(vec_to_state(_s), _a)[0]
    CA_kept = CA[mask]
    CA_ws = np.zeros((CA_kept.shape[0], 2))
    for i in range(CA_kept.shape[0]):
        _a = np.nonzero(CA_kept[i].reshape((nA, -1)))[0][0]
        _s = CA_kept[i].reshape((nA, -1))[_a]
        CA_ws[i, :] = convert_to_ws(vec_to_state(_s), _a)[0]
    X = np.vstack((XA_ws, CA_ws))
    Y = np.hstack((R, G[mask]))
    return LinearRegression().fit(X, Y)
