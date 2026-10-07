# -*- coding: utf-8 -*-
"""实验产物：指标、数据划分、环境、模型、预测、混淆矩阵和错误案例。"""
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import time
import uuid

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.utils import (
    PROJECT_ROOT, RESULTS_DIR, MODELS_DIR, evaluate, file_sha256, write_json,
)

# (结果文件名, 表格方法名, 图表短标签)，汇总与绘图使用同一份定义。
EXPERIMENTS = [
    ("task1_binary_bow", "Binary Bag of Words", "Binary\nBoW"),
    ("task1_word_frequency", "Word Frequency", "Word\nFreq"),
    ("task2_glove", "GloVe (pretrained)", "GloVe"),
    ("task2_word2vec_ag", "Word2Vec (AG News)", "Word2Vec\nAG News"),
    ("task2_word2vec_nyt", "Word2Vec (NYT train)", "Word2Vec\nNYT"),
    ("task3_bert", "BERT fine-tuning", "BERT"),
]
PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib",
    "gensim", "torch", "transformers", "accelerate", "tokenizers",
    "safetensors", "huggingface-hub", "smart_open", "packaging",
)


def environment_info():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}


def portable_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


class Experiment:
    """每次运行使用独立产物目录，成功完成后才更新顶层结果 JSON。"""

    def __init__(self, name, splits, config, *, results_dir=None, models_dir=None):
        if name not in {item[0] for item in EXPERIMENTS}:
            raise ValueError(f"未知实验名称：{name}")
        self.started = time.perf_counter()
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.name, self.splits, self.config = name, splits, config
        self.labels = splits["metadata"]["label_order"]
        self.results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
        model_root = Path(models_dir) if models_dir is not None else MODELS_DIR
        self.detail_dir = self.results_dir / "details" / name / self.run_id
        self.model_dir = model_root / name / self.run_id
        self.detail_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.environment = environment_info()
        self.source_hashes = {p.name: file_sha256(p)
                              for p in sorted((PROJECT_ROOT / "src").glob("*.py"))}
        self.manifest_path = self.results_dir / "splits" / (splits["metadata"]["split_id"] + ".json")
        write_json(self.manifest_path, splits["metadata"])
        write_json(self.detail_dir / "environment.json", self.environment)
        pinned = [f"{name}=={version}" for name, version
                  in self.environment["packages"].items() if version is not None]
        (self.detail_dir / "requirements-resolved.txt").write_text(
            "\n".join(pinned) + "\n", encoding="utf-8")

    def save_model(self, model, filename="model.joblib"):
        path = self.model_dir / filename
        joblib.dump(model, path, compress=3)
        return path

    def _save_predictions(self, split_name, predictions):
        split = self.splits[split_name]
        metrics = evaluate(split["label"], predictions, labels=self.labels)
        frame = pd.DataFrame({
            "source_row": split["source_row"],
            "text": split["text"],
            "true_label": split["label"],
            "predicted_label": list(predictions),
        })
        frame["correct"] = frame["true_label"] == frame["predicted_label"]
        frame.to_csv(self.detail_dir / f"{split_name}_predictions.csv",
                     index=False, encoding="utf-8-sig")
        errors = frame.loc[~frame["correct"]]
        errors.to_csv(self.detail_dir / f"{split_name}_errors.csv",
                      index=False, encoding="utf-8-sig")
        write_json(self.detail_dir / f"{split_name}_error_examples.json",
                   errors.head(20).to_dict(orient="records"))
        report = classification_report(
            split["label"], predictions, labels=self.labels,
            output_dict=True, zero_division=0,
        )
        write_json(self.detail_dir / f"{split_name}_classification_report.json", report)
        matrix = confusion_matrix(split["label"], predictions, labels=self.labels)
        matrix_frame = pd.DataFrame(matrix, index=self.labels, columns=self.labels)
        matrix_frame.index.name = "true_label"
        matrix_frame.to_csv(self.detail_dir / f"{split_name}_confusion_matrix.csv",
                            encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(matrix, display_labels=self.labels).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{self.name}: {split_name}")
        fig.tight_layout()
        fig.savefig(self.detail_dir / f"{split_name}_confusion_matrix.png", dpi=150)
        plt.close(fig)
        return metrics

    def finish(self, val_predictions, test_predictions, *, extra=None):
        val = self._save_predictions("val", val_predictions)
        test = self._save_predictions("test", test_predictions)
        # GloVe 的 classifier.joblib 配合原始预训练向量使用；
        # 其他模型的完整表示对象/权重也在 model_dir 内。
        model_files = [p for p in self.model_dir.rglob("*") if p.is_file()]
        if not model_files:
            raise RuntimeError("模型尚未保存，不能将实验标记为完成。")
        meta = self.splits["metadata"]
        payload = {
            "schema_version": 2,
            "experiment": self.name,
            "run_id": self.run_id,
            "created_at_utc": self.created_at,
            **test,
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "split_id": meta["split_id"],
            "protocol_id": meta["protocol_id"],
            "dataset_sha256": meta["dataset_sha256"],
            "split_manifest": portable_path(self.manifest_path),
            "label_order": self.labels,
            "class_counts": meta["class_counts"],
            "config": self.config,
            "environment": self.environment,
            "source_sha256": self.source_hashes,
            "details_dir": portable_path(self.detail_dir),
            "model_dir": portable_path(self.model_dir),
            "model_files": [portable_path(p) for p in sorted(model_files)],
            "diagnostics": extra or {},
            "total_seconds": round(time.perf_counter() - self.started, 3),
        }
        write_json(self.model_dir / "experiment.json", payload)
        write_json(self.detail_dir / "result.json", payload)
        write_json(self.results_dir / f"{self.name}.json", payload)
        print(f"[{self.name}] Validation: accuracy={val['accuracy']:.4f}, macro_f1={val['macro_f1']:.4f}")
        print(f"[{self.name}] Test: accuracy={test['accuracy']:.4f}, macro_f1={test['macro_f1']:.4f}")
        print(f"模型：{self.model_dir}")
        print(f"预测、混淆矩阵、错误案例：{self.detail_dir}")
        return payload


def load_results(results_dir=None, *, expected_protocol_id=None):
    """拒绝旧格式或不同划分/预处理协议的结果，防止误作横向比较。"""
    root = Path(results_dir) if results_dir is not None else RESULTS_DIR
    records, protocol_ids = {}, set()
    for name, _, _ in EXPERIMENTS:
        path = root / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 2 or not data.get("protocol_id"):
            raise ValueError(f"{path.name} 是缺少实验协议的旧结果，请先归档并重新运行。")
        if data.get("experiment") != name:
            raise ValueError(f"{path.name} 与其中的实验名称不一致。")
        for metric in ("accuracy", "macro_f1", "val_accuracy", "val_macro_f1"):
            value = data.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{path.name} 的 {metric} 无效。")
        protocol_ids.add(data["protocol_id"])
        records[name] = data
    if len(protocol_ids) > 1:
        raise ValueError("结果使用了不同数据划分或分词协议，不能合并；请统一后重新运行。")
    if expected_protocol_id is not None and protocol_ids and protocol_ids != {expected_protocol_id}:
        raise ValueError("已有结果与当前数据/预处理协议不一致，请归档旧结果并重新运行。")
    return records
