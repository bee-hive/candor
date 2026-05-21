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

## Citation

```bibtex
@inproceedings{mandyam2026candor,
  title={CANDOR: Counterfactual ANnotated DOubly Robust Off-Policy Evaluation},
  author={Mandyam, Aishwarya and Tang, Shengpu and Yao, Jiayu and Wiens, Jenna and Engelhardt, Barbara E.},
  booktitle={Conference on Health, Inference, and Learning (CHIL)},
  year={2026}
}
```
