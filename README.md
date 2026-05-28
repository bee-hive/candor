# CANDOR

**CANDOR: Counterfactual ANnotated DOubly Robust Off-Policy Evaluation**

Aishwarya Mandyam, Shengpu Tang, Jiayu Yao, Jenna Wiens, Barbara E. Engelhardt.
Published at **CHIL 2026**.

- Paper (arXiv): https://arxiv.org/abs/2412.08052
- Paper (OpenReview): https://openreview.net/forum?id=yDcg6xgaiO

## About

Off-policy evaluation (OPE) estimates the performance of a target policy before deployment, using data collected under a different behavior policy. CANDOR introduces a family of ``doubly robust''- inspired OPE estimators that incorporate **counterfactual annotations** which are expert-supplied estimates of what the reward would have been under a counterfactual action. The key result is theoretical and empirical evidence that, when the reward model is misspecified, imperfect counterfactual annotations are most useful when fed into the *reward model* component rather than the *importance sampling* component of a doubly-robust estimator.

This repository contains the OPE estimators introduced in the paper.

## Layout

```
src/
  estimators.py                   # shared OPE estimators (IS, DM, DM-IS, IS+, ...)
  toy_bandit/
    data_generation.py            # 2-state contextual bandit data generator
    experiment.py                 # single experiment driver: MSE + bootstrap CIs
  heartsteps/
    data_generation.py            # HeartSteps simulator + on-the-fly data generator
    experiment.py                 # single experiment driver: MSE + bootstrap CIs
  sepsis/
    data_generation.py            # sepsis MDP-based data generation + loading
    experiment.py                 # single experiment driver: MSE + bootstrap CIs
    data_generation/              # supporting sepsisSimDiabetes simulator + priors
```

## Estimators

[`src/estimators.py`](src/estimators.py) defines the OPE estimators (`standard_is`, `direct_method`, `dm_is`, `is+`, `dm_is+`, plus `dm+`/`dm+_is`/`dm+_is+` wrappers) as environment-agnostic per-sample functions. They consume callable reward models and policies.

## Toy bandit

[`src/toy_bandit/`](src/toy_bandit/) is a 2-state contextual bandit used to study how all CANDOR estimators behave under varying annotation bias, annotation variance, and reward-model misspecification.

- [`data_generation.py`](src/toy_bandit/data_generation.py) defines the MDP constants (`R`, `sigma`, `policy_b`, `policy_e`, `d0`, `ww`, `runs`, `N`) and `load_data_train_eval(...)`, which generates a paired (eval, train) dataset. The eval split is what the OPE estimators are computed on; the train split is what the reward model is fit on. Knobs:
  - `bias`, `std`: bias and added std of the counterfactual annotations.
  - `missingness`: fraction of counterfactual annotations available (state 0 only; state 1 never has counterfactual annotations).
  - `policy_b_arr`, `n`: optional overrides for the behavior policy and per-seed sample count.
- [`experiment.py`](src/toy_bandit/experiment.py) exposes one function, `run_experiment(bias, std, ms=False, weight_counterfactual=0.3, missingness=1.0, ...)`, which fits the reward model on the train split (with 50/50 state-label corruption when `ms=True`), runs all nine estimators on the eval split across all seeds, and returns a `DataFrame` with columns `['Approach', 'MSE', 'MSE_lower_bound', 'MSE_higher_bound']`. The CIs are 95% percentile bootstrap intervals over per-seed squared errors.

Example:

```python
from toy_bandit.experiment import run_experiment

df = run_experiment(bias=0.5, std=0.3, ms=True)
print(df)
```

## HeartSteps

[`src/heartsteps/`](src/heartsteps/) is a one-step contextual-bandit version of the HeartSteps physical-activity simulator. Each sample resets the env, so the only randomness across samples is the initial step count `y0` drawn from `d0`.

- [`data_generation.py`](src/heartsteps/data_generation.py) defines the `HeartStepsEnvironment` simulator, the module constants (`runs=50`, `N=200`, `d0`, `policy_b`, `policy_e`, `sigma_R`), `load_data_train_eval(...)` (paired eval/train data, generated on the fly), and `target_policy_value(...)` for the analytic truth. Knobs:
  - `bias`, `std`: counterfactual annotation bias and added std. The CA for the non-taken action is drawn `Normal(true_yc + bias, sigma_R + std)`, matching the convention in `aistats_heartsteps.ipynb`.
  - `missingness`: fraction of counterfactual annotations available.
  - `policy_b_arr`, `n`: optional behavior-policy and sample-size overrides.
- [`experiment.py`](src/heartsteps/experiment.py) exposes `run_experiment(bias, std, ms=False, weight_counterfactual=0.3, missingness=1.0, ...)`, which fits `sklearn` linear-regression reward models on the train split (with `ms=True` randomly flipping `phi(s, a)` ↔ `phi(s, 1-a)` per sample as the misspecification mechanism), runs all nine estimators on the eval split, and returns a `DataFrame` with columns `['Approach', 'MSE', 'MSE_lower_bound', 'MSE_higher_bound']`.

Example:

```python
from heartsteps.experiment import run_experiment

df = run_experiment(bias=0.5, std=0.3, ms=True)
print(df)
```

## Sepsis

[`src/sepsis/`](src/sepsis/) wraps the sepsisSimDiabetes MDP as a one-step contextual bandit (env reset per sample) with 1442 states and 8 actions. Unlike the other two environments, sepsis data is **pre-generated** to `.npy` files because the kitchen-sink bias × variance grid is too expensive to recompute on every experiment run.

### Generating data

Run the full generation pipeline once (this writes per-seed `.npy` files for every behavior policy × bias × variance combination, plus `true_values.npy`):

```python
from sepsis.data_generation import generate_all

generate_all(output_dir='datagen/cb')   # full grid: ~hours
```

For a quick smoke test, pass smaller arguments:

```python
generate_all(
    output_dir='/tmp/sepsis_small',
    behaviorPol_indices=[0],
    bias_values=[0.1],
    variance_values=[0.1],
    n_runs=2,
    nsimsamps=50,
)
```

`generate_all` is composed of three building blocks you can call individually if you want finer control:

- `generate_factual(behaviorPol_idx, output_dir, n_runs, nsimsamps)` — factual `(XA, A, R, NX, X_IDX)` for one behavior policy. (port of cell 9 of the historical data-prep notebook)
- `generate_kitchen_sink_ca(behaviorPol_idx, output_dir, bias_values, variance_values, n_runs)` — counterfactual annotations for every `(bias, variance, seed)`. Requires `generate_factual` to have run first. The CA is drawn `Normal(true_yc + bias, 0.005 + variance)`. (port of cell 18)
- `compute_true_values(output_dir, n_runs)` — per-seed Monte-Carlo estimate of `V^{pi_e}` over all non-absorbing MDP states. (port of cell 24)

### Running experiments

[`data_generation.py`](src/sepsis/data_generation.py) also exposes `load_data(...)` (reads back one seed's data) and the four `learn_rhat*` reward-model fitters. [`experiment.py`](src/sepsis/experiment.py) consumes them via `run_experiment(bias, std, ms=False, weight_counterfactual=0.5, missingness=1.0, behaviorPol_idx=0, data_dir=None)`. Knobs:

- `bias`, `std`: must match a `(bias, variance)` pair that was generated (the loader reads `Ca_bias={bias}_variance={std}.npy`).
- `ms`: when `True`, fit the reward model on the high-dim one-hot state-action features (misspecified); when `False`, fit on the 2-d `convert_to_ws` features (well-specified — slow because each prediction performs an MDP transition).
- `missingness`: applied post-hoc by zeroing a fraction of the loaded CAs (`G` → `nan`).
- `behaviorPol_idx`: 0–5, selecting one of the six behavior policies in `behavior_policies`.
- `data_dir`: override the default `DATA_DIR` (env var `SEPSIS_DATA_DIR`, otherwise `./datagen`).

The returned DataFrame has the same columns as the other two settings. Sepsis runs **eight** estimators rather than nine — the standard `DR_combined` from `estimators.py` is intentionally skipped because the sepsis behavior policies place probability 0 on some actions, which makes the raw counterfactual importance weight `policy_e/policy_b` blow up. The C-IS family sidesteps this via the augmented `policy_b_plus`.

Example:

```python
from sepsis.experiment import run_experiment

df = run_experiment(bias=0.1, std=0.1, ms=True, behaviorPol_idx=0,
                    data_dir='datagen/cb')
print(df)
```

## Citation

```bibtex
@inproceedings{mandyam2026candor,
  title={CANDOR: Counterfactual ANnotated DOubly Robust Off-Policy Evaluation},
  author={Mandyam, Aishwarya and Tang, Shengpu and Yao, Jiayu and Wiens, Jenna and Engelhardt, Barbara E.},
  booktitle={Conference on Health, Inference, and Learning (CHIL)},
  year={2026}
}
```
