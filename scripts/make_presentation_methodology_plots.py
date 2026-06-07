from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-term-ai")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CHECK_MODELS = [
    "B0",
    "B2",
    "B3",
    "H1_local_hybrid",
    "G5_Qwen0p5_G1",
    "G5_Qwen1p5_G1",
    "G4_Qwen_4bit",
    "G3_Qwen",
]

CHECK_LABELS = {
    "B0": "B0\nEmbedding",
    "B2": "B2\nMLP scorer",
    "B3": "B3\nCross-encoder",
    "H1_local_hybrid": "H1\nlocal hybrid",
    "G5_Qwen0p5_G1": "G5 0.5B\nSFT",
    "G5_Qwen1p5_G1": "G5 1.5B\nSFT",
    "G4_Qwen_4bit": "G4 3B\n4-bit",
    "G3_Qwen": "G3 3B\nKD/SFT",
}

METHOD_GROUP = {
    "B0": "Embedding",
    "B2": "Embedding + MLP",
    "B3": "Cross-encoder",
    "H1_local_hybrid": "Hybrid",
    "G5_Qwen0p5_G1": "Compressed LM",
    "G5_Qwen1p5_G1": "Compressed LM",
    "G4_Qwen_4bit": "Quantized LM",
    "G3_Qwen": "3B LM",
}

GROUP_COLORS = {
    "Embedding": "#8A8F98",
    "Embedding + MLP": "#6F747D",
    "Cross-encoder": "#1B8A8F",
    "Hybrid": "#E79A32",
    "Compressed LM": "#3A9D5D",
    "Quantized LM": "#7F63B8",
    "3B LM": "#C84A3A",
    "API": "#C45AA2",
    "Zero-shot LM": "#7EA6D8",
    "SFT/KD LM": "#356EAF",
}

TASK_ORDER = [
    "Raw Meaning Selection",
    "Synonym Selection",
    "Sense Disambiguation",
    "Antonym Selection",
    "Context Cloze",
]

TASK_LABELS = {
    "Raw Meaning Selection": "Raw\nMeaning",
    "Synonym Selection": "Synonym",
    "Sense Disambiguation": "Sense",
    "Antonym Selection": "Antonym",
    "Context Cloze": "Context\nCloze",
}

FINAL_RUNS = {
    "B0": ("runs/B0_test_final/metric_log.json", "Embedding"),
    "B1": ("runs/B1_test_final/metric_log.json", "Embedding + logistic"),
    "B2": ("runs/B2_test_final/metric_log.json", "Embedding + MLP"),
    "B3": ("runs/B3_test_final/metric_log.json", "Cross-encoder"),
    "B4 API": ("runs/B4_test_final/metric_log.json", "API"),
    "H1 hybrid": ("runs/H1_test_final/metric_log.json", "Hybrid"),
    "G0 Qwen": ("runs/G0_Qwen_test_final/metric_log.json", "Zero-shot LM"),
    "G1 Qwen": ("runs/G1_Qwen_test_final/metric_log.json", "SFT/KD LM"),
    "G2 Qwen": ("runs/G2_Qwen_test_final/metric_log.json", "SFT/KD LM"),
    "G3 Qwen": ("runs/G3_Qwen_test_final/metric_log.json", "3B LM"),
    "G4 Qwen 4-bit": ("runs/G4_Qwen_test_final/4bit/metric_log.json", "Quantized LM"),
    "G5 0.5B ZS": ("runs/G5_Qwen0p5_ZS_test_final/metric_log.json", "Zero-shot LM"),
    "G5 0.5B G1": ("runs/G5_Qwen0p5_G1_test_final/metric_log.json", "Compressed LM"),
    "G5 1.5B ZS": ("runs/G5_Qwen1p5_ZS_test_final/metric_log.json", "Zero-shot LM"),
    "G5 1.5B G1": ("runs/G5_Qwen1p5_G1_test_final/metric_log.json", "Compressed LM"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100 * value:.1f}%"


def setup_style() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.family": "Noto Sans CJK JP",
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
        }
    )


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def check_metrics_frame(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in CHECK_MODELS:
        path = root / model / "metric_log.json"
        if not path.exists():
            continue
        d = load_json(path)
        rows.append(
            {
                "model": model,
                "label": CHECK_LABELS[model],
                "group": METHOD_GROUP[model],
                "n": d.get("n"),
                "accuracy": d.get("accuracy"),
                "macro_f1": d.get("macro_f1"),
                "ece": d.get("ece"),
                "brier_score": d.get("brier_score"),
                "latency_p50": d.get("latency_p50"),
                "latency_p95": d.get("latency_p95"),
                "tokens_per_sec": d.get("tokens_per_sec"),
                "peak_mb": d.get("peak_VRAM_or_RAM"),
                "strict_parse_error_rate": d.get("strict_parse_error_rate"),
                "parse_error_rate": d.get("parse_error_rate"),
                "errors": int(round((d.get("n") or 0) * (1 - (d.get("accuracy") or 0)))),
                "task_accuracy": d.get("task_accuracy") or {},
                "task_counts": d.get("task_counts") or {},
                "bootstrap_accuracy_95ci": d.get("bootstrap_accuracy_95ci"),
            }
        )
    return pd.DataFrame(rows)


def final_metrics_frame(cwd: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, (rel_path, group) in FINAL_RUNS.items():
        path = cwd / rel_path
        if not path.exists():
            continue
        d = load_json(path)
        rows.append(
            {
                "model": label,
                "label": label,
                "group": group,
                "n": d.get("n"),
                "accuracy": d.get("accuracy"),
                "macro_f1": d.get("macro_f1"),
                "ece": d.get("ece"),
                "brier_score": d.get("brier_score"),
                "latency_p50": d.get("latency_p50"),
                "latency_p95": d.get("latency_p95"),
                "tokens_per_sec": d.get("tokens_per_sec"),
                "peak_mb": d.get("peak_VRAM_or_RAM"),
                "strict_parse_error_rate": d.get("strict_parse_error_rate"),
                "parse_error_rate": d.get("parse_error_rate"),
                "errors": int(round((d.get("n") or 0) * (1 - (d.get("accuracy") or 0)))),
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy_bars(df: pd.DataFrame, out_dir: Path) -> None:
    df = df.set_index("model").loc[[m for m in CHECK_MODELS if m in set(df["model"])]].reset_index()
    colors = [GROUP_COLORS[g] for g in df["group"]]
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["accuracy"] * 100, color=colors, width=0.72)
    ax.set_title("Check-500: Methodology Contrast by Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 101.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.axhline(95, color="#333333", lw=1, ls="--", alpha=0.35)
    ax.text(len(df) - 0.25, 95.5, "95% deployment-grade band", ha="right", va="bottom", color="#333333")
    for bar, acc, err in zip(bars, df["accuracy"], df["errors"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{acc*100:.1f}%\nerr {err}", ha="center", va="bottom", fontsize=10)
    legend_groups = list(dict.fromkeys(df["group"]))
    ax.legend(handles=[Patch(facecolor=GROUP_COLORS[g], label=g) for g in legend_groups], loc="lower right", frameon=False)
    save_fig(fig, out_dir, "01_check500_methodology_accuracy")


def plot_frontier(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    label_offsets = {
        "B3": (8, 12),
        "H1_local_hybrid": (8, -2),
        "G5_Qwen0p5_G1": (8, -10),
        "G5_Qwen1p5_G1": (8, 8),
        "G4_Qwen_4bit": (8, 8),
        "G3_Qwen": (8, 0),
    }
    for _, row in df.iterrows():
        size = max(120, min(1800, float(row["peak_mb"] or 1) * 0.22))
        ax.scatter(
            row["latency_p50"],
            row["accuracy"] * 100,
            s=size,
            color=GROUP_COLORS[row["group"]],
            alpha=0.82,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.annotate(
            row["label"].replace("\n", " "),
            (row["latency_p50"], row["accuracy"] * 100),
            textcoords="offset points",
            xytext=label_offsets.get(row["model"], (8, 6)),
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Median latency per item, log scale (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 101)
    ax.set_title("Check-500: Accuracy vs Latency vs Memory")
    ax.text(21, 64, "fast but weak", color="#555555")
    ax.text(350, 96.8, "compressed replacement zone", color="#2F7D4D")
    ax.text(1650, 99.2, "highest quality,\nheavy runtime", color="#943629")
    ax.legend(
        handles=[Patch(facecolor=GROUP_COLORS[g], label=g) for g in sorted(set(df["group"]))],
        loc="lower right",
        frameon=False,
        ncol=2,
    )
    save_fig(fig, out_dir, "02_check500_accuracy_latency_memory_frontier")


def plot_task_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    matrix = []
    labels = []
    for model in CHECK_MODELS:
        row = df[df["model"] == model]
        if row.empty:
            continue
        task_accuracy = row.iloc[0]["task_accuracy"]
        matrix.append([task_accuracy.get(task, np.nan) * 100 for task in TASK_ORDER])
        labels.append(CHECK_LABELS[model].replace("\n", " "))
    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_title("Check-500: Task Slice Accuracy Heatmap")
    ax.set_xticks(np.arange(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            text = "n/a" if np.isnan(arr[i, j]) else f"{arr[i, j]:.0f}%"
            color = "white" if not np.isnan(arr[i, j]) and arr[i, j] < 45 else "#111111"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    cbar.set_label("Accuracy (%)")
    save_fig(fig, out_dir, "03_check500_task_slice_heatmap")


def plot_antonym_slice(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "model": row["model"],
                "label": row["label"],
                "group": row["group"],
                "overall": row["accuracy"],
                "antonym": row["task_accuracy"].get("Antonym Selection", np.nan),
            }
        )
    hard = pd.DataFrame(rows)
    hard = hard.set_index("model").loc[[m for m in CHECK_MODELS if m in set(hard["model"])]].reset_index()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(hard))
    bars = ax.bar(x, hard["antonym"] * 100, color=[GROUP_COLORS[g] for g in hard["group"]], width=0.68)
    ax.plot(x, hard["overall"] * 100, color="#222222", marker="o", lw=2.2, label="Overall accuracy")
    ax.set_title("Check-500: Hard Slice Reveals Model Gap")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 104)
    ax.set_xticks(x)
    ax.set_xticklabels(hard["label"])
    ax.text(0.02, 0.92, "Antonym Selection n=14", transform=ax.transAxes, fontsize=12, color="#555555")
    for bar, val in zip(bars, hard["antonym"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.legend(loc="upper left", frameon=False)
    save_fig(fig, out_dir, "04_check500_antonym_hard_slice")


def plot_stage_ladder(df: pd.DataFrame, out_dir: Path) -> None:
    sequence = ["B0", "B2", "B3", "G5_Qwen0p5_G1", "G5_Qwen1p5_G1", "G4_Qwen_4bit", "G3_Qwen"]
    sub = df.set_index("model").loc[[m for m in sequence if m in set(df["model"])]].reset_index()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(sub))
    ax.plot(x, sub["accuracy"] * 100, color="#1B4F72", marker="o", lw=3, ms=9)
    ax.fill_between(x, sub["accuracy"] * 100, 55, color="#1B4F72", alpha=0.08)
    ax.set_title("Check-500: Methodology Stage Ladder")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 101.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "similarity\nbaseline",
            "embedding\nclassifier",
            "pairwise\nreranker",
            "0.5B local\nstudent",
            "1.5B local\nstudent",
            "3B 4bit\nruntime",
            "3B KD/SFT\nquality ceiling",
        ]
    )
    for i, row in sub.iterrows():
        ax.text(i, row["accuracy"] * 100 + 0.9, f"{row['accuracy']*100:.1f}%", ha="center", fontsize=11)
    ax.annotate(
        "+34.4pp",
        xy=(2, sub.loc[sub["model"] == "B3", "accuracy"].iloc[0] * 100),
        xytext=(0.45, 88),
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"},
        fontsize=13,
        color="#333333",
    )
    ax.annotate(
        "best quality\nslowest runtime",
        xy=(len(sub) - 1, sub.iloc[-1]["accuracy"] * 100),
        xytext=(len(sub) - 2.3, 98.7),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#943629"},
        fontsize=12,
        color="#943629",
    )
    save_fig(fig, out_dir, "05_check500_methodology_stage_ladder")


def plot_pairwise_deltas(root: Path, out_dir: Path) -> None:
    report_dir = root / "reports"
    rows = []
    for path in sorted(report_dir.glob("*.json")):
        d = load_json(path)
        name = path.stem
        delta_a_minus_b = d.get("paired_bootstrap_accuracy_delta", {}).get("delta")
        ci = d.get("paired_bootstrap_accuracy_delta", {}).get("ci") or [None, None]
        if delta_a_minus_b is None:
            continue
        rows.append(
            {
                "comparison": name.replace("_vs_", " vs ").replace("_", " "),
                "delta": -float(delta_a_minus_b) * 100,
                "lo": -float(ci[1]) * 100,
                "hi": -float(ci[0]) * 100,
                "p": d.get("mcnemar", {}).get("p_value"),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    y = np.arange(len(df))
    xerr = np.vstack([df["delta"] - df["lo"], df["hi"] - df["delta"]])
    ax.errorbar(df["delta"], y, xerr=xerr, fmt="o", color="#1B4F72", ecolor="#6B7C93", elinewidth=2, capsize=4)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["comparison"])
    ax.set_xlabel("Second model accuracy - first model accuracy (percentage point)")
    ax.set_title("Check-500: paired accuracy delta and bootstrap 95% CI")
    for i, row in df.iterrows():
        p_text = "p<0.001" if row["p"] is not None and row["p"] < 0.001 else (f"p={row['p']:.3f}" if row["p"] is not None else "")
        ax.text(row["hi"] + 0.7, i, f"{row['delta']:+.1f}pp  {p_text}", va="center", fontsize=9)
    save_fig(fig, out_dir, "06_check500_pairwise_delta_ci")


def plot_calibration_contract(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df.set_index("model").loc[[m for m in CHECK_MODELS if m in set(df["model"])]].reset_index()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(sub))
    w = 0.36
    ax.bar(x - w / 2, sub["ece"] * 100, width=w, color="#5B8DB8", label="ECE")
    ax.bar(x + w / 2, sub["strict_parse_error_rate"] * 100, width=w, color="#B85B5B", label="strict parsing error")
    ax.set_title("Check-500: Calibration and Output Contract")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 95)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["label"])
    ax.legend(loc="upper left", frameon=False)
    ax.annotate(
        "G3 is accurate and well calibrated,\nbut often violates strict JSON contract",
        xy=(len(sub) - 1 + w / 2, sub.iloc[-1]["strict_parse_error_rate"] * 100),
        xytext=(4.6, 72),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#8D3333"},
        fontsize=12,
        color="#8D3333",
    )
    save_fig(fig, out_dir, "07_check500_calibration_contract")


def plot_hybrid_routing(root: Path, df: pd.DataFrame, out_dir: Path) -> None:
    path = root / "H1_local_hybrid" / "prediction_log.jsonl"
    if not path.exists():
        return
    rows = read_jsonl(path)
    counts = Counter(row.get("hybrid_reason", "unknown") for row in rows)
    labels = ["high_confidence", "mid_confidence_cross_encoder", "low_confidence_fallback"]
    values = [counts.get(label, 0) for label in labels]
    colors = ["#8A8F98", "#1B8A8F", "#3A9D5D"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 7.5), gridspec_kw={"width_ratios": [1.0, 1.4]})
    ax1.pie(values, labels=None, autopct="%1.1f%%", colors=colors, startangle=110, textprops={"fontsize": 10})
    ax1.set_title("Hybrid routing")
    ax1.legend(
        [
            Patch(facecolor=color)
            for color in colors
        ],
        [
            f"{reason_label}: {value} items"
            for reason_label, value in zip(["high confidence", "mid confidence: cross-encoder", "low confidence: fallback"], values, strict=True)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        fontsize=9,
    )
    sub = df[df["model"].isin(["B0", "B3", "H1_local_hybrid", "G5_Qwen0p5_G1"])].copy()
    order = ["B0", "B3", "H1_local_hybrid", "G5_Qwen0p5_G1"]
    sub = sub.set_index("model").loc[order].reset_index()
    x = np.arange(len(sub))
    ax2.bar(x, sub["accuracy"] * 100, color=[GROUP_COLORS[g] for g in sub["group"]])
    ax2.set_title("Hybrid result and components")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(55, 101)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub["label"])
    for i, row in sub.iterrows():
        ax2.text(i, row["accuracy"] * 100 + 0.8, f"{row['accuracy']*100:.1f}%", ha="center")
    fig.suptitle("Check-500: Local Hybrid mostly uses Cross-encoder route", fontsize=18, y=0.98)
    save_fig(fig, out_dir, "08_check500_hybrid_routing")


def plot_error_overlap(root: Path, out_dir: Path) -> None:
    predictions: dict[str, dict[str, bool]] = {}
    for model in CHECK_MODELS:
        path = root / model / "prediction_log.jsonl"
        if not path.exists():
            continue
        predictions[model] = {row["item_id"]: row.get("prediction") != row.get("label") for row in read_jsonl(path)}
    models = [m for m in CHECK_MODELS if m in predictions]
    arr = np.zeros((len(models), len(models)), dtype=float)
    for i, a in enumerate(models):
        a_errors = {item for item, wrong in predictions[a].items() if wrong}
        for j, b in enumerate(models):
            b_errors = {item for item, wrong in predictions[b].items() if wrong}
            arr[i, j] = len(a_errors & b_errors)
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    im = ax.imshow(arr, cmap="Blues", aspect="auto")
    ax.set_title("Check-500: Error Overlap Matrix")
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([CHECK_LABELS[m].replace("\n", " ") for m in models], rotation=35, ha="right")
    ax.set_yticklabels([CHECK_LABELS[m].replace("\n", " ") for m in models])
    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(j, i, f"{int(arr[i, j])}", ha="center", va="center", color="#111111", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    cbar.set_label("both-wrong item count")
    save_fig(fig, out_dir, "09_check500_error_overlap")


def plot_finaltest_methodology(final_df: pd.DataFrame, out_dir: Path) -> None:
    wanted = ["B0", "B1", "B2", "B3", "B4 API", "H1 hybrid", "G3 Qwen", "G5 0.5B G1", "G5 1.5B G1", "G4 Qwen 4-bit"]
    sub = final_df[final_df["model"].isin(wanted)].copy().set_index("model").loc[[m for m in wanted if m in set(final_df["model"])]].reset_index()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(sub))
    colors = [GROUP_COLORS.get(g, "#999999") for g in sub["group"]]
    ax.bar(x, sub["accuracy"] * 100, color=colors)
    ax.set_title("Original final test: Methodology Accuracy Landscape")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 102.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["label"], rotation=25, ha="right")
    for i, row in sub.iterrows():
        ax.text(i, row["accuracy"] * 100 + 0.7, f"{row['accuracy']*100:.1f}%\nn={int(row['n'])}", ha="center", fontsize=9)
    legend_groups = list(dict.fromkeys(sub["group"]))
    ax.legend(handles=[Patch(facecolor=GROUP_COLORS.get(g, "#999999"), label=g) for g in legend_groups], loc="lower right", frameon=False, ncol=2)
    save_fig(fig, out_dir, "10_finaltest_original_methodology_accuracy")


def plot_finaltest_lm_stage(final_df: pd.DataFrame, out_dir: Path) -> None:
    wanted = ["G0 Qwen", "G1 Qwen", "G2 Qwen", "G3 Qwen", "G5 0.5B ZS", "G5 0.5B G1", "G5 1.5B ZS", "G5 1.5B G1", "G4 Qwen 4-bit"]
    sub = final_df[final_df["model"].isin(wanted)].copy().set_index("model").loc[[m for m in wanted if m in set(final_df["model"])]].reset_index()
    fig, ax1 = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(sub))
    ax1.bar(x, sub["accuracy"] * 100, color=[GROUP_COLORS.get(g, "#999999") for g in sub["group"]], alpha=0.88, label="Accuracy")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(98.5, 100.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["label"], rotation=25, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, sub["ece"] * 100, color="#B85B5B", marker="o", lw=2.5, label="ECE")
    ax2.set_ylabel("ECE (%)")
    ax2.set_ylim(0, max(5, float(sub["ece"].max() * 115)))
    ax1.set_title("Original final test: LM stages show reliability more than accuracy")
    for i, row in sub.iterrows():
        ax1.text(i, row["accuracy"] * 100 + 0.02, f"{row['accuracy']*100:.2f}%", ha="center", fontsize=8)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)
    save_fig(fig, out_dir, "11_finaltest_lm_stage_accuracy_ece")


def plot_finaltest_compression(final_df: pd.DataFrame, out_dir: Path) -> None:
    wanted = ["G3 Qwen", "G5 0.5B G1", "G5 1.5B G1", "G4 Qwen 4-bit"]
    sub = final_df[final_df["model"].isin(wanted)].copy()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    label_offsets = {
        "G3 Qwen": (12, 6),
        "G5 0.5B G1": (10, 4),
        "G5 1.5B G1": (10, 10),
        "G4 Qwen 4-bit": (12, 6),
    }
    for _, row in sub.iterrows():
        size = max(180, min(1800, float(row["peak_mb"] or 1) * 0.22))
        ax.scatter(row["latency_p50"], row["accuracy"] * 100, s=size, color=GROUP_COLORS.get(row["group"], "#999999"), alpha=0.82, edgecolor="white", linewidth=1.5)
        ax.annotate(row["label"], (row["latency_p50"], row["accuracy"] * 100), textcoords="offset points", xytext=label_offsets.get(row["model"], (8, 6)), fontsize=10)
    ax.set_xscale("log")
    ax.set_title("Original final test: Compression Boundary")
    ax.set_xlabel("Median latency per item, log scale (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(99.65, 100.05)
    ax.text(270, 99.925, "student models:\nnear-3B quality,\nlower runtime", color="#2F7D4D", fontsize=12)
    ax.text(1450, 99.735, "4bit: lower latency than 3B fp16,\ncalibration needs recheck", color="#5E4690", fontsize=12)
    save_fig(fig, out_dir, "12_finaltest_compression_boundary")


def plot_final_vs_check(check_df: pd.DataFrame, final_df: pd.DataFrame, out_dir: Path) -> None:
    pairs = [
        ("B0", "B0"),
        ("B2", "B2"),
        ("B3", "B3"),
        ("G5_Qwen0p5_G1", "G5 0.5B G1"),
        ("G5_Qwen1p5_G1", "G5 1.5B G1"),
        ("G4_Qwen_4bit", "G4 Qwen 4-bit"),
        ("G3_Qwen", "G3 Qwen"),
    ]
    rows = []
    for check_model, final_model in pairs:
        c = check_df[check_df["model"] == check_model]
        f = final_df[final_df["model"] == final_model]
        if c.empty or f.empty:
            continue
        rows.append(
            {
                "label": CHECK_LABELS[check_model].replace("\n", " "),
                "check": c.iloc[0]["accuracy"],
                "final": f.iloc[0]["accuracy"],
                "group": c.iloc[0]["group"],
            }
        )
    sub = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(sub))
    w = 0.38
    ax.bar(x - w / 2, sub["final"] * 100, width=w, color="#B8C2CC", label="Original final raw test")
    ax.bar(x + w / 2, sub["check"] * 100, width=w, color=[GROUP_COLORS[g] for g in sub["group"]], label="Check-500 mixed/hard set")
    ax.set_title("Final raw test vs Check-500: hard slice reveals methodology gap")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 101.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["label"], rotation=25, ha="right")
    ax.legend(loc="lower right", frameon=False)
    for i, row in sub.iterrows():
        ax.text(i - w / 2, row["final"] * 100 + 0.5, f"{row['final']*100:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, row["check"] * 100 + 0.5, f"{row['check']*100:.1f}", ha="center", fontsize=8)
    save_fig(fig, out_dir, "13_final_vs_check500_accuracy")


def plot_runtime_steps(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df.set_index("model").loc[[m for m in CHECK_MODELS if m in set(df["model"])]].reset_index()
    fig, ax1 = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(sub))
    total_latency_sec = sub["latency_p50"] * sub["n"] / 1000
    ax1.bar(x, total_latency_sec, color=[GROUP_COLORS[g] for g in sub["group"]], alpha=0.88)
    ax1.set_ylabel("Estimated runtime for 500 items from p50 latency (sec)")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["label"])
    ax1.set_title("Check-500: Runtime Cost by Method")
    for i, value in enumerate(total_latency_sec):
        label = f"{value:.0f}s" if value < 90 else f"{value/60:.1f}m"
        ax1.text(i, value * 1.12, label, ha="center", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(x, sub["accuracy"] * 100, color="#222222", marker="o", lw=2.2, label="Accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(55, 101)
    ax2.legend(loc="upper left", frameon=False)
    save_fig(fig, out_dir, "14_check500_runtime_cost_by_method")


def write_summary_tables(check_df: pd.DataFrame, final_df: pd.DataFrame, out_dir: Path) -> None:
    flat = check_df.drop(columns=["task_accuracy", "task_counts", "bootstrap_accuracy_95ci"], errors="ignore")
    flat.to_csv(out_dir / "check500_metrics_summary.csv", index=False)
    final_df.to_csv(out_dir / "finaltest_selected_metrics_summary.csv", index=False)
    task_rows = []
    for _, row in check_df.iterrows():
        for task, acc in row["task_accuracy"].items():
            task_rows.append({"model": row["model"], "label": row["label"].replace("\n", " "), "task": task, "accuracy": acc})
    pd.DataFrame(task_rows).to_csv(out_dir / "check500_task_accuracy.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PPT-friendly methodology contrast plots from term_ai experiment outputs.")
    parser.add_argument("--check-root", default="runs/no_api_no_retrain_check")
    parser.add_argument("--out-dir", default="reports/presentation_plots")
    args = parser.parse_args()

    cwd = Path.cwd()
    check_root = Path(args.check_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    check_df = check_metrics_frame(check_root)
    if check_df.empty:
        raise SystemExit(f"no check metrics found under {check_root}")
    final_df = final_metrics_frame(cwd)

    write_summary_tables(check_df, final_df, out_dir)
    plot_accuracy_bars(check_df, out_dir)
    plot_frontier(check_df, out_dir)
    plot_task_heatmap(check_df, out_dir)
    plot_antonym_slice(check_df, out_dir)
    plot_stage_ladder(check_df, out_dir)
    plot_pairwise_deltas(check_root, out_dir)
    plot_calibration_contract(check_df, out_dir)
    plot_hybrid_routing(check_root, check_df, out_dir)
    plot_error_overlap(check_root, out_dir)
    if not final_df.empty:
        plot_finaltest_methodology(final_df, out_dir)
        plot_finaltest_lm_stage(final_df, out_dir)
        plot_finaltest_compression(final_df, out_dir)
        plot_final_vs_check(check_df, final_df, out_dir)
    plot_runtime_steps(check_df, out_dir)

    generated = sorted(path.name for path in out_dir.glob("*.png"))
    print(json.dumps({"out_dir": str(out_dir), "png_count": len(generated), "png_files": generated}, indent=2))


if __name__ == "__main__":
    main()
