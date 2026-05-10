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
```

## Estimators

[`src/estimators.py`](src/estimators.py) defines the OPE estimators (`standard_is`, `direct_method`, `dm_is`, `is+`, `dm_is+`, plus `dm+`/`dm+_is`/`dm+_is+` wrappers) as environment-agnostic per-sample functions. They consume callable reward models and policies.

## Citation

```bibtex
@inproceedings{mandyam2026candor,
  title={CANDOR: Counterfactual ANnotated DOubly Robust Off-Policy Evaluation},
  author={Mandyam, Aishwarya and Tang, Shengpu and Yao, Jiayu and Wiens, Jenna and Engelhardt, Barbara E.},
  booktitle={Conference on Health, Inference, and Learning (CHIL)},
  year={2026}
}
```
