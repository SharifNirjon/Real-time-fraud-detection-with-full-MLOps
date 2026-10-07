"""Report figures (PR curves, calibration, feature importance, SHAP)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

PALETTE = ["#2a6fdb", "#e0731f", "#2f9e44", "#c92a2a", "#7048e8", "#868e96"]


def pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (name, s), c in zip(scores.items(), PALETTE, strict=False):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=c, lw=1.8, label=f"{name} (AP={average_precision_score(y, s):.3f})")
    ax.axhline(np.mean(y), color="#adb5bd", ls="--", lw=1, label=f"base rate {np.mean(y):.3f}")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall on TEST",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def calibration(curves: dict[str, dict], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="#adb5bd", ls="--", lw=1, label="perfect")
    for (name, cv), c in zip(curves.items(), PALETTE, strict=False):
        ax.plot(
            cv["curve"]["mean_pred"],
            cv["curve"]["frac_pos"],
            "o-",
            color=c,
            lw=1.6,
            ms=4,
            label=f"{name} (Brier={cv['brier']:.4f})",
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed fraud rate",
        title="Calibration on TEST (quantile bins)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def feature_importance(names: list[str], gain: np.ndarray, out: Path, top: int = 25) -> Path:
    idx = np.argsort(gain)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 7))
    ax.barh([names[i] for i in idx], gain[idx] / gain.sum(), color=PALETTE[0])
    ax.set(xlabel="Share of total gain", title=f"LightGBM top-{top} features (gain)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def shap_summary(booster, X: np.ndarray, names: list[str], out: Path) -> Path:
    import shap

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]
    plt.figure()
    shap.summary_plot(values, X, feature_names=names, max_display=20, show=False)
    plt.title("SHAP summary (TEST sample)")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close("all")
    return out
