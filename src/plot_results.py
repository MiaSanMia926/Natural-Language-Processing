# -*- coding: utf-8 -*-
"""绘制同一实验协议下的 Test Accuracy / Macro-F1。"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from src.experiment import EXPERIMENTS, load_results
from src.utils import PROJECT_ROOT, RESULTS_DIR, load_nyt_splits, write_json


def load_available_results(*, results_dir=None, expected_protocol_id=None):
    records = load_results(results_dir, expected_protocol_id=expected_protocol_id)
    labels, accuracy, macro_f1 = [], [], []
    for name, _, label in EXPERIMENTS:
        if name in records:
            labels.append(label)
            accuracy.append(records[name]["accuracy"])
            macro_f1.append(records[name]["macro_f1"])
    return labels, accuracy, macro_f1


def main():
    protocol = load_nyt_splits(verbose=False)["metadata"]["protocol_id"]
    records = load_results(expected_protocol_id=protocol)
    labels, accuracy, macro_f1 = load_available_results(expected_protocol_id=protocol)
    if not labels:
        print("尚无当前实验协议的结果，请先运行任务脚本。")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#334155",
                         "text.color": "#172B4D", "svg.fonttype": "none"})
    names = [name for name, _, _ in EXPERIMENTS if name in records]
    display = [label.replace("\n", " ") for name, _, label in EXPERIMENTS if name in records]
    colors = ["#287C8E", "#CE7736"]

    def save(fig, stem):
        for extension in ("png", "pdf", "svg"):
            fig.savefig(RESULTS_DIR / f"{stem}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.0), layout="constrained")
    y = np.arange(len(names))
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("A  Accuracy", "B  Macro-F1")):
        for i in y:
            if i % 2 == 0:
                ax.axhspan(i-.45, i+.45, color="#F1F5F9", zorder=0)
        for offset, prefix, color, label in ((-.12, "val_", colors[0], "Validation"), (.12, "", colors[1], "Test")):
            values = [records[n][prefix+metric] for n in names]
            ax.scatter(values, y+offset, s=52, color=color, label=label, zorder=3)
            for xx, yy in zip(values, y+offset):
                ax.annotate(f"{xx*100:.2f}", (xx, yy), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
        ax.set_yticks(y, display)
        ax.invert_yaxis()
        ax.set_xlim((.958, 1.003) if metric=="accuracy" else (.906, .983))
        ax.set_title(title, loc="left", pad=14)
        ax.set_xlabel("Score (zoomed axis; labels in %)")
        ax.grid(axis="x", color="#CBD5E1", alpha=.6)
    axes[1].set_yticklabels([])
    axes[0].legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0,-.18))
    fig.suptitle("NYT text classification | six methods", fontsize=15, fontweight="bold")
    save(fig, "comparison_chart")

    reports = [json.loads((PROJECT_ROOT / records[n]["details_dir"] / "test_classification_report.json").read_text(encoding="utf-8")) for n in names]
    classes = records[names[0]]["label_order"]
    scores = np.array([[r[c]["f1-score"] for c in classes] for r in reports])
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    im = ax.imshow(scores, cmap="YlGnBu", vmin=.85, vmax=1, aspect="auto")
    for i in range(len(names)):
        for j in range(len(classes)):
            ax.text(j,i,f"{scores[i,j]*100:.2f}%",ha="center",va="center",color="white" if scores[i,j]>.955 else "#172B4D")
    ax.set_yticks(y, display); ax.set_xticks(range(3), classes)
    ax.set_title("Test F1 by class", loc="left", pad=14)
    fig.colorbar(im, ax=ax, label="F1 score", shrink=.85)
    save(fig,"class_f1_chart")

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
    for ax,n,label in zip(axes.flat,names,display):
        matrix = pd.read_csv(PROJECT_ROOT / records[n]["details_dir"] / "test_confusion_matrix.csv",index_col=0).loc[classes,classes].to_numpy()
        normalized = matrix / matrix.sum(axis=1,keepdims=True)
        ax.imshow(normalized,cmap="Blues",vmin=0,vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f"{matrix[i,j]}\n{normalized[i,j]*100:.1f}%",ha="center",va="center",fontsize=9,color="white" if normalized[i,j]>.5 else "#172B4D")
        ax.set_xticks(range(3),classes,rotation=20); ax.set_yticks(range(3),classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(label)
    fig.suptitle("Test confusion matrices | count and row percentage",fontsize=15,fontweight="bold")
    save(fig,"confusion_matrices")

    if "task3_bert" in records:
        history=json.loads((PROJECT_ROOT / records["task3_bert"]["details_dir"] / "training_log.json").read_text())
        train=[r for r in history if "loss" in r]; val=[r for r in history if "eval_loss" in r]
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout="constrained")
        axes[0].plot([r['epoch'] for r in train],[r['loss'] for r in train],color=colors[0],label="Training (logged intervals)")
        axes[0].plot([r['epoch'] for r in val],[r['eval_loss'] for r in val],"o-",color=colors[1],label="Validation (epoch end)")
        axes[0].set(title="A  Loss",xlabel="Epoch",ylabel="Cross-entropy"); axes[0].legend(fontsize=8,frameon=False)
        for key,color,label in (("eval_accuracy",colors[0],"Validation Accuracy"),("eval_macro_f1",colors[1],"Validation Macro-F1")):
            axes[1].plot([r['epoch'] for r in val],[r[key] for r in val],"o-",color=color,label=label)
        axes[1].set(title="B  Validation scores",xlabel="Epoch",ylabel="Score",ylim=(.93,.99),xticks=[1,2,3]); axes[1].legend(fontsize=8,frameon=False)
        for ax in axes: ax.grid(alpha=.2)
        save(fig,"bert_training_curve")
    output = RESULTS_DIR / "comparison_chart.png"
    write_json(RESULTS_DIR / "comparison_chart_metadata.json", {
        "protocol_id": protocol,
        "runs": {name: data["run_id"] for name, data in records.items()},
    })
    print(f"图表：{output}；已包含 {len(labels)}/{len(EXPERIMENTS)} 组实验。")


if __name__ == "__main__":
    main()
