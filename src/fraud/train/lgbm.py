"""LightGBM training helpers: early stopping on valid PR-AUC, weighting variants, Optuna."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np

log = logging.getLogger(__name__)


def base_params(seed: int, learning_rate: float) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "average_precision",  # = PR-AUC, used for early stopping
        "learning_rate": learning_rate,
        "num_leaves": 63,
        "min_child_samples": 100,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "seed": seed,
        "num_threads": 0,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def fit(
    params: dict[str, Any],
    dtrain: lgb.Dataset,
    dvalid: lgb.Dataset,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[lgb.Booster, float]:
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return booster, float(booster.best_score["valid"]["average_precision"])


def weighting_variants(y: np.ndarray) -> dict[str, dict[str, Any]]:
    ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    return {
        "none": {},
        "scale_pos_weight": {"scale_pos_weight": ratio},
        "sqrt_scale_pos_weight": {"scale_pos_weight": float(np.sqrt(ratio))},
    }


def optuna_search(
    base: dict[str, Any],
    dtrain: lgb.Dataset,
    dvalid: lgb.Dataset,
    n_trials: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        p = dict(base)
        p.update(
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 500, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.2, 0.9),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
            min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        )
        booster, score = fit(p, dtrain, dvalid, num_boost_round, early_stopping_rounds)
        trial.set_user_attr("best_iteration", booster.best_iteration)
        log.info(
            "optuna trial %d: PR-AUC=%.4f iters=%d", trial.number, score, booster.best_iteration
        )
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(
        {
            k: base[k]
            for k in [
                "num_leaves",
                "min_child_samples",
                "feature_fraction",
                "bagging_fraction",
                "lambda_l2",
            ]
        }
        | {"min_gain_to_split": 0.0}
    )
    study.optimize(objective, n_trials=n_trials)
    best = dict(base) | study.best_params
    history = [{"trial": t.number, "pr_auc": t.value, **t.params} for t in study.trials]
    return best, history
